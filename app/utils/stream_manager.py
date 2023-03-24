from asyncio import sleep
import json
import re
from typing import AsyncGenerator
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
import httpx
from app.models import UserChatGptMessage
from app.utils.logger import logger


class AsyncStream:
    user_messages: dict[list[dict[str, str]]] = {}
    user_contexts: dict[str, dict[str, list[dict[str, str]] | dict[str, str]]] = {}

    @staticmethod
    async def hello_world() -> AsyncGenerator:
        for char in "Hello World! This is Test Streaming Text.":
            yield char
            await sleep(0.05)

    @staticmethod
    async def chatgpt(
        user_message_histories: list[dict[str, str]],
        api_key: str,
        timeout: float,
        gpt_role: str = "assistant",
    ) -> AsyncGenerator:
        chat_gpt_response: str = ""
        data_regex_pattern: re.Pattern = re.compile(r"^data: \s*")
        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                try:
                    async with client.stream(
                        method="POST",
                        url="https://api.openai.com/v1/chat/completions",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                        json={
                            "model": "gpt-3.5-turbo",
                            "messages": user_message_histories,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "n": 1,
                            "stream": True,
                            "presence_penalty": 0,
                            "frequency_penalty": 0,
                        },
                    ) as streaming_response:
                        if streaming_response.status_code != 200:
                            logger.error(streaming_response.status_code)
                            raise HTTPException(
                                status_code=streaming_response.status_code,
                                detail="OpenAI 서버 응답오류",
                            )
                        async for chunked_text in streaming_response.aiter_text():
                            chunked_data: list[str] = [
                                data_regex_pattern.sub("", data)
                                for data in chunked_text.split("\n\n")
                                if data.startswith("data: ")
                            ]
                            for data in chunked_data:
                                if data == "[DONE]":
                                    user_message_histories.append(
                                        {
                                            "role": gpt_role,
                                            "content": chat_gpt_response,
                                        }
                                    )
                                    logger.info(msg=user_message_histories)
                                    break
                                char: str = json.loads(data)["choices"][0]["delta"].get(
                                    "content", ""
                                )
                                if char == "" or char == "\n\n":
                                    continue
                                else:
                                    chat_gpt_response += char
                                    yield char
                except httpx.TimeoutException as exception:
                    logger.error(msg=exception)
                    continue
                else:
                    break


class ChatGptCommands:
    @staticmethod
    def not_existing_callback(*args, context: dict) -> str:
        return f"{context['profile']['user_id']}님, 죄송합니다. 현재 그런 명령어는 지원하지 않습니다."

    @staticmethod
    def clear(*args, context: dict) -> str:
        response: str = f"좋아요. {context['profile']['user_id']}님과의 {len(context['message_histories'])}개의  기억을 삭제할게요."
        context["message_histories"].clear()
        return response

    @staticmethod
    def test(*args, context: dict) -> str:
        return f"args: {args} | context: {context}"

    @staticmethod
    def system(*args, context: dict) -> str:
        if len(args) < 1:
            return "/system SYSTEM_MESSAGE와 같은 형식으로 입력해야 합니다."
        system_message: str = " ".join(args)
        context["message_histories"].append(
            {
                "role": "system",
                "content": system_message,
            }
        )
        return f"시스템 메시지를 `{system_message}`로 추가하였습니다!"


async def establish_websocket_connection(
    websocket: WebSocket,
    user_id: str,
    api_key: str,
):
    user_context: dict[
        str, list[dict[str, str]] | dict[str, str]
    ] = AsyncStream.user_contexts[user_id]
    prompt_in_progress: bool = False
    while True:
        try:
            user_chat_gpt_message: UserChatGptMessage = UserChatGptMessage.parse_raw(
                await websocket.receive_text()
            )
            user_message: str = user_chat_gpt_message.user_message
            if user_message.startswith("/"):
                user_command: list = user_message.split()
                callback_name: str = user_command[0][1:]
                callback_args: list = user_command[1:]
                callback_response: str = (
                    getattr(ChatGptCommands, callback_name)(
                        *callback_args, context=user_context
                    )
                    if hasattr(ChatGptCommands, callback_name)
                    else ChatGptCommands.not_existing_callback(context=user_context)
                )
                await websocket.send_text("\n\n")
                await websocket.send_text(callback_response)
                await websocket.send_text("\n\n\n")
                continue
            user_context["message_histories"].append(
                {
                    "role": user_context["profile"]["user_role"],
                    "content": user_message,
                }
            )
            prompt_in_progress = True
            await websocket.send_text("\n\n")
            async for char in AsyncStream.chatgpt(
                user_message_histories=user_context["message_histories"],
                api_key=api_key,
                timeout=5.0,
                gpt_role=user_context["profile"]["gpt_role"],
            ):  # AsyncGenerator
                await websocket.send_text(char)
            await websocket.send_text("\n\n\n")
            prompt_in_progress = False
        except (RuntimeError, WebSocketDisconnect):
            user_context["message_histories"].pop() if prompt_in_progress else ...
            break
