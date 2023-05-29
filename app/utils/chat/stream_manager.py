import asyncio

from fastapi import WebSocket, WebSocketDisconnect
from orjson import JSONDecodeError
from orjson import loads as orjson_loads
from pydantic import ValidationError
from app.database.schemas.auth import Users

from app.errors.chat_exceptions import (
    ChatException,
    ChatInterruptedException,
    ChatOtherException,
    ChatTextGenerationException,
    ChatTooMuchTokenException,
)
from app.models.chat_models import ChatRoles, UserChatProfile
from app.models.llms import LLMModels
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.cache_manager import CacheManager
from app.utils.chat.chat_commands import (
    command_handler,
    create_new_chat_room,
)
from app.utils.chat.message_handler import MessageHandler
from app.utils.chat.message_manager import MessageManager
from app.utils.chat.vectorstore_manager import VectorStoreManager
from app.utils.chat.websocket_manager import SendToWebsocket
from app.utils.logger import api_logger
from app.viewmodels.base_models import MessageFromWebsocket, SummarizedResult


async def initialize_callback(user_id: str) -> list[UserChatProfile]:
    user_chat_profiles: list[UserChatProfile] = await CacheManager.fetch_chat_profiles(user_id=user_id)
    if not user_chat_profiles:
        # create new chatroom
        return [(await create_new_chat_room(user_id=user_id)).user_chat_profile]
    else:
        # get latest chatroom
        user_chat_profiles.sort(key=lambda profile: profile.created_at, reverse=True)
        return user_chat_profiles


async def harvest_done_tasks(buffer: BufferedUserContext) -> None:
    """
    This function checks and handles the done tasks in buffer.task_list.
    :param buffer: BufferedUserContext object
    :return: None
    """
    harvested_tasks = set(task for task in buffer.task_list if task.done() and not task.cancelled())
    update_tasks = []

    for task in harvested_tasks:
        try:
            task_result = task.result()
            if isinstance(task_result, SummarizedResult):
                context_index = buffer.find_index_of_chatroom(task_result.chat_room_id)
                if context_index is None:
                    continue
                role = ChatRoles.get_member(task_result.role)
                context = await buffer._sorted_ctxts.at(context_index)
                message_history_index = await buffer.find_index_of_message_history(
                    user_chat_context=context,
                    role=role,
                    message_history_uuid=task_result.uuid,
                )
                if message_history_index is None:
                    continue

                update_tasks.append(
                    MessageManager.set_message_history_safely(
                        user_chat_context=context,
                        summarized_content=task_result.content,
                        index=message_history_index,
                        role=role,
                    )
                )
        except Exception as e:
            api_logger.exception(f"Some error occurred while harvesting done tasks: {e}")

    if update_tasks:
        api_logger.info(f"Running update tasks: {update_tasks}")

    try:
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                api_logger.exception(f"Some error occurred while running update tasks: {result}")
    except Exception as e:
        api_logger.exception(f"Unexpected error occurred while running update tasks: {e}")
    finally:
        buffer.task_list = [task for task in buffer.task_list if task not in harvested_tasks]


class ChatStreamManager:
    @classmethod
    async def begin_chat(cls, websocket: WebSocket, user: Users) -> None:
        # initialize variables
        buffer: BufferedUserContext = BufferedUserContext(
            user=user,
            websocket=websocket,
            initialize_callback=initialize_callback,
            read_callback=CacheManager.read_context_from_profile,
        )
        await buffer.init()
        await SendToWebsocket.init(
            buffer=buffer,
            send_chat_rooms=True,
            send_previous_chats=True,
            send_models=True,
            send_selected_model=True,
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
                                send_previous_chats=True,
                            )
                        elif "model" in received_json:
                            found_model = LLMModels._member_map_.get(received_json["model"])
                            if found_model is not None:
                                buffer.current_user_chat_context.llm_model = found_model  # type: ignore
                                await SendToWebsocket.init(buffer=buffer, send_selected_model=True)
                                await CacheManager.update_model(user_chat_context=buffer.current_user_chat_context)
            elif received_bytes is not None:
                await buffer.queue.put(
                    await VectorStoreManager.embed_file_to_vectorstore(
                        file=received_bytes, filename=filename, collection_name=buffer.current_user_chat_context.user_id
                    )
                )

    @classmethod
    async def _websocket_sender(cls, buffer: BufferedUserContext) -> None:
        while True:  # loop until connection is closed
            try:
                await SendToWebsocket.init(
                    buffer=buffer,
                    send_tokens=True,
                )

                await harvest_done_tasks(buffer=buffer)
                item: MessageFromWebsocket | str = await buffer.queue.get()
                await harvest_done_tasks(buffer=buffer)

                if isinstance(item, str):
                    await SendToWebsocket.message(
                        websocket=buffer.websocket,
                        msg=item,
                        chat_room_id=buffer.current_chat_room_id,
                    )
                elif isinstance(item, MessageFromWebsocket):
                    if item.chat_room_id != buffer.current_chat_room_id:
                        # This is a message from another chat room, interpreted as change of context, ignoring message
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
                        buffer.last_user_message = item.msg
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
                send_chat_rooms=True,
                send_previous_chats=True,
            )
        else:
            # if received chat_room_id is in chat_room_ids, get context from memory
            await buffer.change_context_to(index=index)
            await SendToWebsocket.init(
                buffer=buffer,
                send_previous_chats=True,
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
