from asyncio import sleep
import json

import re
from typing import AsyncGenerator
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
import httpx

# from app.common.config import OPENAI_API_KEY
from app.errors.exceptions import Responses_500
from app.models import UserChatGptMessage
from app.utils.logger import logger


class AsyncStream:
    context: dict[list[dict[str, str]]] = {}

    @staticmethod
    async def hello_world() -> AsyncGenerator:
        for char in "Hello World! This is Test Streaming Text.":
            yield char
            await sleep(0.05)

    @staticmethod
    async def chatgpt(
        websocket: WebSocket,
        context: list[dict[str, str]],
        api_key: str,
        timeout: float,
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
                            "messages": context,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "n": 1,
                            "stream": True,
                            "presence_penalty": 0,
                            "frequency_penalty": 0,
                        },
                    ) as streaming_response:
                        if streaming_response.status_code != 200:
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
                                    context.append(
                                        {
                                            "role": "assistant",
                                            "content": chat_gpt_response,
                                        }
                                    )
                                    logger.info(msg=context)
                                    break
                                char: str = json.loads(data)["choices"][0]["delta"].get(
                                    "content", ""
                                )
                                if char == "" or char == "\n\n":
                                    continue
                                else:
                                    # print("CHAR:", char)
                                    chat_gpt_response += char
                                    yield char
                except httpx.TimeoutException as exception:
                    logger.error(msg=exception)
                    continue
                else:
                    break


async def establish_websocket_connection(
    websocket: WebSocket,
    user_id: str,
    api_key: str,
):
    user_context: list[dict[str, str]] = AsyncStream.context.get(user_id)
    prompt_in_progress: bool = False
    while True:
        try:
            user_chat_gpt_message = UserChatGptMessage.parse_raw(
                await websocket.receive_text()
            )
            user_context.append(
                {
                    "role": "user",
                    "content": user_chat_gpt_message.user_message,
                }
            )
            prompt_in_progress = True
            await websocket.send_text("\n\n")
            async for char in AsyncStream.chatgpt(
                context=user_context,
                api_key=api_key,
                timeout=5.0,
                websocket=websocket,
            ):  # AsyncGenerator
                await websocket.send_text(char)
            await websocket.send_text("\n\n\n")
            prompt_in_progress = False
        except (RuntimeError, WebSocketDisconnect):
            user_context.pop() if prompt_in_progress else ...
            break
