from fastapi import WebSocket, WebSocketDisconnect

from app.errors.gpt_exceptions import GptOtherException, GptTextGenerationException, GptTooMuchTokenException
from app.utils.chatgpt.chatgpt_cache_manager import chatgpt_cache_manager
from app.utils.chatgpt.chatgpt_commands import (
    ChatGptCommands,
    ResponseType,
    create_new_chat_room,
    get_contexts_sorted_from_recent_to_past,
)
from app.utils.chatgpt.chatgpt_message_manager import MessageManager
from app.utils.chatgpt.chatgpt_websocket_manager import HandleMessage, SendToWebsocket
from app.utils.logger import api_logger
from app.utils.chatgpt.chatgpt_buffer import BufferedUserContext
from app.viewmodels.base_models import MessageFromWebsocket, MessageToWebsocket
from app.viewmodels.gpt_models import GptRoles


async def command_handler(
    callback_name: str,
    callback_args: list[str],
    received: MessageFromWebsocket,
    websocket: WebSocket,
    buffer: BufferedUserContext,
    openai_api_key: str,
):
    callback_response, response_type = await ChatGptCommands._get_command_response(
        callback_name=callback_name,
        callback_args=callback_args,
        buffer=buffer,
    )
    if response_type is ResponseType.DO_NOTHING:
        return
    elif response_type is ResponseType.HANDLE_GPT:
        await HandleMessage.gpt(
            translate=received.translate,
            openai_api_key=openai_api_key,
            buffer=buffer,
        )
        return
    elif response_type is ResponseType.HANDLE_USER:
        await HandleMessage.user(
            msg=callback_response,
            translate=received.translate,
            buffer=buffer,
        )
        return
    elif response_type is ResponseType.HANDLE_BOTH:
        await HandleMessage.user(
            msg=callback_response,
            translate=received.translate,
            buffer=buffer,
        )
        await HandleMessage.gpt(
            translate=received.translate,
            openai_api_key=openai_api_key,
            buffer=buffer,
        )
        return
    elif response_type is ResponseType.REPEAT_COMMAND:
        await command_handler(
            callback_name=callback_name,
            callback_args=callback_args,
            received=received,
            websocket=websocket,
            buffer=buffer,
            openai_api_key=openai_api_key,
        )


async def begin_chat(
    websocket: WebSocket,
    user_id: str,
    openai_api_key: str,
) -> None:
    # initialize variables
    buffer: BufferedUserContext = BufferedUserContext(
        user_id=user_id,
        websocket=websocket,
        sorted_ctxts=await get_contexts_sorted_from_recent_to_past(
            user_id=user_id,
            chat_room_ids=await chatgpt_cache_manager.get_all_chat_rooms(user_id=user_id),
        ),
    )
    await SendToWebsocket.initiation_of_chat(websocket=websocket, buffer=buffer)

    while True:  # loop until connection is closed
        try:
            # receive message from websocket
            received: MessageFromWebsocket = MessageFromWebsocket.parse_raw(await websocket.receive_text())

            if received.chat_room_id != buffer.current_chat_room_id:  # change chat room
                index: int | None = buffer.find_index_of_chatroom(received.chat_room_id)
                if index is None:
                    # if received chat_room_id is not in chat_room_ids, create new chat room
                    await create_new_chat_room(
                        user_id=user_id,
                        new_chat_room_id=received.chat_room_id,
                        buffer=buffer,
                    )
                    buffer.change_context_to(index=0)
                else:
                    # if received chat_room_id is in chat_room_ids, get context from memory
                    buffer.change_context_to(index=index)
                await SendToWebsocket.initiation_of_chat(websocket=websocket, buffer=buffer)

            if received.msg.startswith("/"):  # if user message is command
                splitted: list[str] = received.msg[1:].split(" ")
                await command_handler(
                    callback_name=splitted[0],
                    callback_args=splitted[1:],
                    received=received,
                    websocket=websocket,
                    buffer=buffer,
                    openai_api_key=openai_api_key,
                )
                continue

            await HandleMessage.user(
                msg=received.msg,
                translate=received.translate,
                buffer=buffer,
            )
            await HandleMessage.gpt(
                translate=received.translate,
                openai_api_key=openai_api_key,
                buffer=buffer,
            )

        except WebSocketDisconnect:
            raise WebSocketDisconnect(code=1000, reason="client disconnected")
        except GptTextGenerationException:
            await MessageManager.rpop_message_history_safely(
                user_gpt_context=buffer.current_user_gpt_context, role=GptRoles.USER
            )
        except GptOtherException:
            await MessageManager.rpop_message_history_safely(
                user_gpt_context=buffer.current_user_gpt_context, role=GptRoles.USER
            )
            await MessageManager.rpop_message_history_safely(
                user_gpt_context=buffer.current_user_gpt_context, role=GptRoles.GPT
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
