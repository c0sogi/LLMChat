import asyncio

from fastapi import WebSocket, WebSocketDisconnect
from orjson import JSONDecodeError
from orjson import loads as orjson_loads
from pydantic import ValidationError

from app.errors.chat_exceptions import (
    ChatException,
    ChatInterruptedException,
    ChatOtherException,
    ChatTextGenerationException,
    ChatTooMuchTokenException,
)
from app.models.chat_models import ChatRoles
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.cache_manager import CacheManager
from app.utils.chat.chat_commands import (
    command_handler,
    create_new_chat_room,
    get_contexts_sorted_from_recent_to_past,
)
from app.utils.chat.message_handler import MessageHandler
from app.utils.chat.message_manager import MessageManager
from app.utils.chat.vectorstore_manager import VectorStoreManager
from app.utils.chat.websocket_manager import SendToWebsocket
from app.utils.logger import api_logger
from app.viewmodels.base_models import MessageFromWebsocket


class ChatStreamManager:
    @classmethod
    async def begin_chat(cls, websocket: WebSocket, user_id: str) -> None:
        # initialize variables
        buffer: BufferedUserContext = BufferedUserContext(
            user_id=user_id,
            websocket=websocket,
            sorted_ctxts=await get_contexts_sorted_from_recent_to_past(
                user_id=user_id,
                chat_room_ids=await CacheManager.get_all_chat_rooms(user_id=user_id),
            ),
        )
        await SendToWebsocket.init(
            buffer=buffer,
        )
        try:
            await asyncio.gather(
                cls._websocket_receiver(buffer=buffer),
                cls._websocket_sender(buffer=buffer),
            )
        except (ChatOtherException, ChatTextGenerationException, ChatTooMuchTokenException) as e:
            api_logger.error(e)
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg="An error occurred. Please try again.",
                chat_room_id=buffer.current_chat_room_id,
            )
        except WebSocketDisconnect:
            return
        except RuntimeError:
            return
        except Exception as e:
            api_logger.error(f"Exception in chat: {e}", exc_info=True)
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg="Internal server error. Please try again.",
                chat_room_id=buffer.current_chat_room_id,
            )

    @staticmethod
    async def _websocket_receiver(buffer: BufferedUserContext) -> None:
        filename: str = ""
        while True:  # loop until connection is closed
            rcvd = await buffer.websocket.receive()
            received_text: str | None = rcvd.get("text")
            received_bytes: bytes | None = rcvd.get("bytes")

            if received_text is not None:
                try:
                    received_json: dict = orjson_loads(received_text)
                    assert isinstance(received_json, dict)
                except (JSONDecodeError, AssertionError):
                    if received_text == "stop":
                        buffer.done.set()
                else:
                    try:
                        await buffer.queue.put(MessageFromWebsocket(**received_json))
                    except ValidationError:
                        if "filename" in received_json:
                            # if received json has filename, it is a file
                            filename = received_json["filename"]
                        elif "chat_room_name" in received_json:
                            buffer.current_user_chat_context.user_chat_profile.chat_room_name = received_json[
                                "chat_room_name"
                            ][:20]
                            await CacheManager.update_profile(user_chat_context=buffer.current_user_chat_context)
                            await SendToWebsocket.init(
                                buffer=buffer,
                                send_previous_chats=False,
                                init_callback=False,
                            )
            elif received_bytes is not None:
                await buffer.queue.put(
                    await VectorStoreManager.embed_file_to_vectorstore(file=received_bytes, filename=filename)
                )

    @classmethod
    async def _websocket_sender(cls, buffer: BufferedUserContext) -> None:
        while True:  # loop until connection is closed
            try:
                item: MessageFromWebsocket | str = await buffer.queue.get()

                if isinstance(item, str):
                    await SendToWebsocket.message(
                        websocket=buffer.websocket,
                        msg=item,
                        chat_room_id=buffer.current_chat_room_id,
                    )
                elif isinstance(item, MessageFromWebsocket):
                    if item.chat_room_id != buffer.current_chat_room_id:
                        # This is a message from another chat room, interpreted as change of context, while ignoring message
                        await cls._change_context(
                            buffer=buffer,
                            changed_chat_room_id=item.chat_room_id,
                        )
                    elif item.msg.startswith("/"):
                        # if user message is command, handle command
                        splitted: list[str] = item.msg[1:].split(" ")
                        await command_handler(
                            callback_name=splitted[0],
                            callback_args=splitted[1:],
                            translate=item.translate,
                            buffer=buffer,
                        )
                    else:
                        await MessageHandler.user(
                            msg=item.msg,
                            translate=item.translate,
                            buffer=buffer,
                        )
                        await MessageHandler.ai(
                            translate=item.translate,
                            buffer=buffer,
                        )
            except ChatException as chat_exception:
                await cls._chat_exception_handler(buffer=buffer, chat_exception=chat_exception)

    @staticmethod
    async def _change_context(buffer: BufferedUserContext, changed_chat_room_id: str) -> None:
        index: int | None = buffer.find_index_of_chatroom(changed_chat_room_id)
        if index is None:
            # if received chat_room_id is not in chat_room_ids, create new chat room
            await create_new_chat_room(
                user_id=buffer.user_id,
                new_chat_room_id=changed_chat_room_id,
                buffer=buffer,
            )
            await SendToWebsocket.init(
                buffer=buffer,
                send_previous_chats=False,
            )
        else:
            # if received chat_room_id is in chat_room_ids, get context from memory
            buffer.change_context_to(index=index)
            await SendToWebsocket.init(
                buffer=buffer,
                send_chat_rooms=False,
            )

    @staticmethod
    async def _chat_exception_handler(buffer: BufferedUserContext, chat_exception: ChatException):
        if isinstance(chat_exception, ChatTextGenerationException):
            await asyncio.gather(
                SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=f"\n\nAn error occurred while generating text: **{chat_exception.msg}**",
                    chat_room_id=buffer.current_chat_room_id,
                    finish=True,
                    model_name=buffer.current_user_chat_context.llm_model.value.name,
                ),
                MessageManager.pop_message_history_safely(
                    user_chat_context=buffer.current_user_chat_context,
                    role=ChatRoles.USER,
                ),
            )
        elif isinstance(chat_exception, ChatInterruptedException):
            await MessageManager.add_message_history_safely(
                user_chat_context=buffer.current_user_chat_context,
                role=ChatRoles.AI,
                content=str(chat_exception.msg),
            )
        elif isinstance(chat_exception, ChatOtherException):
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=str(chat_exception.msg),
                chat_room_id=buffer.current_chat_room_id,
            )
        elif isinstance(chat_exception, ChatTooMuchTokenException):
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=str(chat_exception.msg),
                chat_room_id=buffer.current_user_chat_context.chat_room_id,
            )  # send too much token exception message to websocket
