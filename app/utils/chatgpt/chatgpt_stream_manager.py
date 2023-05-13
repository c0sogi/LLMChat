import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from app.errors.gpt_exceptions import GptOtherException, GptTextGenerationException, GptTooMuchTokenException
from app.utils.chatgpt.chatgpt_buffer import BufferedUserContext
from app.utils.chatgpt.chatgpt_cache_manager import ChatGptCacheManager
from app.utils.chatgpt.chatgpt_commands import (
    command_handler,
    create_new_chat_room,
    get_contexts_sorted_from_recent_to_past,
)
from app.utils.chatgpt.chatgpt_fileloader import read_bytes_to_text
from app.utils.chatgpt.chatgpt_message_manager import MessageManager
from app.utils.chatgpt.chatgpt_vectorstore_manager import VectorStoreManager
from app.utils.chatgpt.chatgpt_websocket_manager import HandleMessage, SendToWebsocket
from app.utils.logger import api_logger
from app.viewmodels.base_models import MessageFromWebsocket, MessageToWebsocket
from app.models.gpt_models import GptRoles


async def begin_chat(
    websocket: WebSocket,
    user_id: str,
) -> None:
    # initialize variables
    buffer: BufferedUserContext = BufferedUserContext(
        user_id=user_id,
        websocket=websocket,
        sorted_ctxts=await get_contexts_sorted_from_recent_to_past(
            user_id=user_id,
            chat_room_ids=await ChatGptCacheManager.get_all_chat_rooms(user_id=user_id),
        ),
    )

    await SendToWebsocket.initiation_of_chat(
        buffer=buffer,
        send_chat_room_ids=True,
        send_previous_chats=True,
    )

    while True:  # loop until connection is closed
        try:
            rcvd: dict = await websocket.receive_json()
            assert isinstance(rcvd, dict)
            if "filename" in rcvd:
                # if user uploads file, embed it
                text: str = await run_in_threadpool(
                    read_bytes_to_text, await websocket.receive_bytes(), rcvd["filename"]
                )
                docs: list[str] = await VectorStoreManager.create_documents(text)
                await SendToWebsocket.message(
                    websocket=websocket,
                    msg=f"Successfully embedded documents. You uploaded file begins with...\n\n```{docs[0][:50]}```...",
                    chat_room_id=buffer.current_chat_room_id,
                )
                continue
            received: MessageFromWebsocket = MessageFromWebsocket(**rcvd)

            if received.chat_room_id != buffer.current_chat_room_id:
                # This is a message from another chat room, interpreted as change of context, while ignoring message
                index: int | None = buffer.find_index_of_chatroom(received.chat_room_id)
                if index is None:
                    # if received chat_room_id is not in chat_room_ids, create new chat room
                    await create_new_chat_room(
                        user_id=user_id,
                        new_chat_room_id=received.chat_room_id,
                        buffer=buffer,
                    )
                    await SendToWebsocket.initiation_of_chat(
                        buffer=buffer,
                        send_previous_chats=False,
                        send_chat_room_ids=True,
                    )
                else:
                    # if received chat_room_id is in chat_room_ids, get context from memory
                    buffer.change_context_to(index=index)
                    await SendToWebsocket.initiation_of_chat(
                        buffer=buffer,
                        send_previous_chats=True,
                        send_chat_room_ids=False,
                    )
                continue

            if received.msg.startswith("/"):
                # if user message is command, handle command
                splitted: list[str] = received.msg[1:].split(" ")
                await command_handler(
                    callback_name=splitted[0],
                    callback_args=splitted[1:],
                    received=received,
                    websocket=websocket,
                    buffer=buffer,
                )
                continue

            await HandleMessage.user(
                msg=received.msg,
                translate=received.translate,
                buffer=buffer,
            )
            await HandleMessage.gpt(
                translate=received.translate,
                buffer=buffer,
            )

        except WebSocketDisconnect:
            raise WebSocketDisconnect(code=1000, reason="client disconnected")
        except (AssertionError, ValidationError):
            await SendToWebsocket.message(
                websocket=websocket,
                msg="Invalid message. Message is not in the correct format, maybe frontend - backend version mismatch?",
                chat_room_id=buffer.current_chat_room_id,
            )
            continue
        except ValueError as e:
            api_logger.error(e, exc_info=True)
            await SendToWebsocket.message(
                websocket=websocket,
                msg="Invalid file type.",
                chat_room_id=buffer.current_chat_room_id,
            )
            continue
        except GptTextGenerationException:
            await asyncio.gather(
                SendToWebsocket.message(
                    websocket=websocket,
                    msg="Text generation failure. Please try again.",
                    chat_room_id=buffer.current_chat_room_id,
                ),
                MessageManager.pop_message_history_safely(
                    user_gpt_context=buffer.current_user_gpt_context,
                    role=GptRoles.USER,
                ),
            )
        except GptOtherException:
            await asyncio.gather(
                SendToWebsocket.message(
                    websocket=websocket,
                    msg="Something's wrong. Please try again.",
                    chat_room_id=buffer.current_chat_room_id,
                ),
                MessageManager.pop_message_history_safely(
                    user_gpt_context=buffer.current_user_gpt_context,
                    role=GptRoles.USER,
                ),
                MessageManager.pop_message_history_safely(
                    user_gpt_context=buffer.current_user_gpt_context,
                    role=GptRoles.GPT,
                ),
            )
        except GptTooMuchTokenException as too_much_token_exception:  # if user message is too long
            await SendToWebsocket.message(
                websocket=websocket,
                msg=too_much_token_exception.msg if too_much_token_exception.msg is not None else "",
                chat_room_id=buffer.current_user_gpt_context.chat_room_id,
            )  # send too much token exception message to websocket
            continue
        except Exception as exception:  # if other exception is raised
            api_logger.error(f"chat exception: {exception}", exc_info=True)
            await websocket.send_json(  # finish stream message
                MessageToWebsocket(
                    msg="Internal Server Error",
                    finish=True,
                    chat_room_id=buffer.current_user_gpt_context.chat_room_id,
                    is_user=False,
                ).dict()
            )
            break
