from typing import AsyncGenerator
from fastapi import WebSocket

import orjson

from app.errors.gpt_exceptions import GptOtherException, GptTextGenerationException, GptTooMuchTokenException
from app.utils.api.translate import Translator
from app.utils.chatgpt.chatgpt_buffer import BufferedUserContext
from app.utils.chatgpt.chatgpt_generation import generate_from_openai, message_history_organizer
from app.utils.chatgpt.chatgpt_message_manager import MessageManager
from app.viewmodels.base_models import MessageToWebsocket
from app.viewmodels.gpt_models import GptRoles, UserGptContext


class SendToWebsocket:
    @staticmethod
    async def initiation_of_chat(
        websocket: WebSocket,
        buffer: BufferedUserContext,
    ) -> None:
        await SendToWebsocket.message(
            websocket=websocket,
            msg=orjson.dumps(
                {
                    "previous_chats": message_history_organizer(
                        user_gpt_context=buffer.current_user_gpt_context,
                        send_to_openai=False,
                    ),
                    "chat_room_ids": buffer.sorted_chat_room_ids,
                }
            ).decode("utf-8"),
            chat_room_id=buffer.current_chat_room_id,
            init=True,
        )

    @staticmethod
    async def message(
        websocket: WebSocket,
        msg: str,
        chat_room_id: str,
        finish: bool = True,
        is_user: bool = False,
        init: bool = False,
    ) -> None:  # send whole message to websocket
        await websocket.send_json(  # send stream message
            MessageToWebsocket(
                msg=msg,
                finish=finish,
                chat_room_id=chat_room_id,
                is_user=is_user,
                init=init,
            ).dict()
        )

    @staticmethod
    async def stream(
        websocket: WebSocket,
        stream: AsyncGenerator,
        chat_room_id: str,
        finish: bool = True,
        is_user: bool = False,
        chunk_size: int = 3,
    ) -> str:  # send whole stream to websocket
        response: str = ""
        chunk_buffer: list[str] = []
        async for delta in stream:  # stream from api
            response += delta
            chunk_buffer.append(delta)
            if len(chunk_buffer) >= chunk_size:
                await websocket.send_json(
                    MessageToWebsocket(
                        msg="".join(chunk_buffer),
                        finish=False,
                        chat_room_id=chat_room_id,
                        is_user=is_user,
                    ).dict()
                )
                chunk_buffer.clear()
        if len(chunk_buffer) > 0:
            # flush remaining chunks
            await websocket.send_json(
                MessageToWebsocket(
                    msg="".join(chunk_buffer),
                    finish=False,
                    chat_room_id=chat_room_id,
                    is_user=is_user,
                ).dict()
            )
        if finish:
            await websocket.send_json(  # finish stream message
                MessageToWebsocket(
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
        msg: str,
        translate: bool,
        buffer: BufferedUserContext,
    ) -> None:
        if translate:  # if user message is translated
            msg = await Translator.auto_translate(
                text=msg,
                src_lang="ko",
                trg_lang="en",
            )
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"## 번역된 질문\n\n{msg}\n\n## 생성된 답변\n\n",
                chat_room_id=buffer.current_chat_room_id,
                finish=False,
            )
        user_token: int = len(buffer.current_user_gpt_context.tokenize(msg))
        if user_token > buffer.current_user_gpt_context.token_per_request:  # if user message is too long
            raise GptTooMuchTokenException(
                msg=f"Message too long. Now {user_token} tokens, but {buffer.current_user_gpt_context.token_per_request} tokens allowed."
            )
        await MessageManager.add_message_history_safely(
            user_gpt_context=buffer.current_user_gpt_context,
            content=msg,
            role=GptRoles.USER,
        )

    @staticmethod
    async def gpt(
        translate: bool,
        openai_api_key: str,
        buffer: BufferedUserContext,
    ) -> None:
        try:
            msg: str = await SendToWebsocket.stream(
                websocket=buffer.websocket,
                chat_room_id=buffer.current_chat_room_id,
                stream=generate_from_openai(
                    user_gpt_context=buffer.current_user_gpt_context,
                    openai_api_key=openai_api_key,
                ),
                finish=False if translate else True,
            )
        except Exception:
            raise GptTextGenerationException(msg="텍스트를 생성하는데 문제가 발생했습니다.")
        try:
            if translate:  # if user message is translated
                translated_msg = await Translator.auto_translate(
                    text=msg,
                    src_lang="en",
                    trg_lang="ko",
                )
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=f"\n\n## 번역된 답변\n\n{translated_msg}",
                    chat_room_id=buffer.current_chat_room_id,
                )
        except Exception:
            raise GptOtherException(msg="번역하는데 문제가 발생했습니다.")
