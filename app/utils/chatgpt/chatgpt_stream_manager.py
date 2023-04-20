from typing import AsyncGenerator, Callable
from inspect import iscoroutinefunction
from fastapi import WebSocket, WebSocketDisconnect
from app.common.config import GOOGLE_TRANSLATE_API_KEY
from app.errors.gpt_exceptions import (
    GptTooMuchTokenException,
)
from app.viewmodels.base_models import SendChatMessage, ReceiveChatMessage
from app.viewmodels.gpt_models import UserGptContext
from app.utils.chatgpt.chatgpt_commands import ChatGptCommands
from app.utils.chatgpt.chatgpt_context_manager import context_manager
from app.utils.chatgpt.chatgpt_generation import generate_from_openai
from app.utils.api.translate import google_translate_api


async def begin_chat(
    websocket: WebSocket,
    user_id: str,
    openai_api_key: str,
) -> None:  # websocket for chat gpt
    user_gpt_context: UserGptContext = context_manager.read_context(user_id)
    while True:  # loop until connection is closed
        try:
            # initialize variables
            received: ReceiveChatMessage = ReceiveChatMessage.parse_raw(await websocket.receive_text())
            msg: str = received.msg
            translate: bool = received.translate
            chat_room_id: int = received.chat_room_id

            # TODO: validate if chat room belongs to user
            # print("received_chat_message: ", msg)

            if msg.startswith("/"):  # if user message is command
                await SendToWebsocket.message(
                    websocket=websocket,
                    msg=await get_command_response(msg=msg, user_gpt_context=user_gpt_context),
                    chat_room_id=chat_room_id,
                )  # send callback response to websocket
                continue
            await HandleMessage.user(
                websocket=websocket,
                user_gpt_context=user_gpt_context,
                msg=msg,
                translate=translate,
                chat_room_id=chat_room_id,
            )
            await HandleMessage.gpt(
                websocket=websocket,
                user_gpt_context=user_gpt_context,
                translate=translate,
                chat_room_id=chat_room_id,
                openai_api_key=openai_api_key,
            )

        except WebSocketDisconnect:
            raise WebSocketDisconnect(code=1000, reason="client disconnected")
        except GptTooMuchTokenException as too_much_token_exception:  # if user message is too long
            await SendToWebsocket.message(
                websocket=websocket,
                msg=too_much_token_exception.msg,
                chat_room_id=chat_room_id,
            )  # send too much token exception message to websocket
            continue
        except Exception as exception:  # if other exception is raised
            print(f"chat exception: {exception}")
            await websocket.send_json(  # finish stream message
                SendChatMessage(
                    msg="",
                    finish=True,
                    chat_room_id=chat_room_id,
                    is_user=False,
                ).dict()
            )
            break


class SendToWebsocket:
    @staticmethod
    async def message(
        websocket: WebSocket,
        msg: str,
        chat_room_id: int = 0,
        finish: bool = True,
        is_user: bool = False,
    ) -> None:  # send whole message to websocket
        await websocket.send_json(  # send stream message
            SendChatMessage(
                msg=msg,
                finish=finish,
                chat_room_id=chat_room_id,
                is_user=is_user,
            ).dict()
        )

    @staticmethod
    async def stream(
        websocket: WebSocket,
        stream: AsyncGenerator,
        chat_room_id: int = 0,
        finish: bool = True,
        is_user: bool = False,
    ) -> str:  # send whole stream to websocket
        response: str = ""
        async for delta in stream:  # stream from api
            response += delta
            await websocket.send_json(
                SendChatMessage(
                    msg=delta,
                    finish=False,
                    chat_room_id=chat_room_id,
                    is_user=is_user,
                ).dict()
            )
        if finish:
            await websocket.send_json(  # finish stream message
                SendChatMessage(
                    msg="",
                    finish=True,
                    chat_room_id=chat_room_id,
                    is_user=is_user,
                ).dict()
            )
        return response


class HandleMessage:
    @staticmethod
    async def user(
        websocket: WebSocket,
        user_gpt_context: UserGptContext,
        msg: str,
        translate: bool,
        chat_room_id: int,
    ) -> None:
        if translate:  # if user message is translated
            translated_msg = await google_translate_api(
                text=msg,
                src_lang="ko",
                trg_lang="en",
                api_key=GOOGLE_TRANSLATE_API_KEY,
            )
            await SendToWebsocket.message(
                websocket=websocket,
                msg=f"[번역된 질문]\n{translated_msg}",
                chat_room_id=chat_room_id,
            )

        user_token: int = len(user_gpt_context.tokenize(translated_msg if translate else msg))
        if user_token > user_gpt_context.token_per_request:  # if user message is too long
            raise GptTooMuchTokenException(
                msg=f"메시지가 너무 길어요. 현재 토큰 개수는 {user_token}로, {user_gpt_context.token_per_request} 이하여야 합니다."
            )
        await user_gpt_context.add_user_message_history_safely(translated_msg if translate else msg)
        context_manager.update_message_histories(
            user_id=user_gpt_context.user_gpt_profile.user_id,
            role="user",
            message_histories=user_gpt_context.user_message_histories,
        )

    @staticmethod
    async def gpt(
        websocket: WebSocket,
        user_gpt_context: UserGptContext,
        translate: bool,
        chat_room_id: int,
        openai_api_key: str,
    ) -> None:
        msg: str = await SendToWebsocket.stream(
            websocket=websocket,
            stream=generate_from_openai(
                user_gpt_context=user_gpt_context,
                openai_api_key=openai_api_key,
            ),
        )
        if translate:  # if user message is translated
            translated_msg = await google_translate_api(
                text=msg,
                src_lang="en",
                trg_lang="ko",
                api_key=GOOGLE_TRANSLATE_API_KEY,
            )
            await SendToWebsocket.message(
                websocket=websocket,
                msg=f"[번역된 답변]\n{translated_msg}",
                chat_room_id=chat_room_id,
            )


async def get_command_response(msg: str, user_gpt_context: UserGptContext) -> str:
    user_command: list = msg.split()
    callback: Callable[[list, UserGptContext], str] = (
        getattr(ChatGptCommands, user_command[0][1:])
        if hasattr(ChatGptCommands, user_command[0][1:])
        else ChatGptCommands.not_existing_callback
    )  # get callback function
    callback_args: list = user_command[1:]  # command args
    if iscoroutinefunction(callback):  # if callback is coroutine function
        callback_response: str = await callback(*callback_args, user_gpt_context=user_gpt_context)
    else:
        callback_response: str = callback(*callback_args, user_gpt_context=user_gpt_context)
    return callback_response
