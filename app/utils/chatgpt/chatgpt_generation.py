from asyncio import sleep
from itertools import zip_longest
from typing import AsyncGenerator
import orjson
import httpx
from app.errors.gpt_exceptions import (
    GptConnectionException,
    GptContentFilterException,
    GptException,
    GptLengthException,
)
from app.utils.chatgpt.chatgpt_message_manager import MessageManager
from app.viewmodels.base_models import SendInitToWebsocket, SendToOpenAI
from app.viewmodels.gpt_models import GptRoles, UserGptContext
from app.utils.chatgpt.chatgpt_config import ChatGPTConfig
from app.utils.logger import api_logger


def message_history_organizer(
    user_gpt_context: UserGptContext, send_to_openai: bool = True
) -> list[dict]:  # organize message history for openai api
    message_histories: list[dict] = []
    if send_to_openai:
        for system_history in user_gpt_context.system_message_histories:
            message_histories.append(SendToOpenAI.from_orm(system_history).dict())  # append system message history
    for user_message_history, gpt_message_history in zip_longest(
        user_gpt_context.user_message_histories,
        user_gpt_context.gpt_message_histories,
    ):
        message_histories.append(
            SendToOpenAI.from_orm(user_message_history).dict()
            if send_to_openai
            else SendInitToWebsocket.from_orm(user_message_history).dict()
        ) if user_message_history is not None else ...  # append user message history
        message_histories.append(
            SendToOpenAI.from_orm(gpt_message_history).dict()
            if send_to_openai
            else SendInitToWebsocket.from_orm(gpt_message_history).dict()
        ) if gpt_message_history is not None else ...  # append gpt message history
    if user_gpt_context.optional_info.get("is_discontinued", False):
        for message_history in reversed(message_histories):
            if message_history["role"] == user_gpt_context.user_gpt_profile.gpt_role:
                message_history["content"] += "...[CONTINUATION]"
                break
    return message_histories  # return message histories to be used in openai api


async def generate_from_openai(
    openai_api_key: str,  # api key for openai
    user_gpt_context: UserGptContext,  # gpt context for user
) -> AsyncGenerator:  # async generator for streaming
    async with httpx.AsyncClient(timeout=ChatGPTConfig.wait_for_timeout) as client:  # initialize client
        is_appending_discontinued_message: bool = False
        content_buffer: str = ""
        while True:  # stream until connection is closed
            if not user_gpt_context.optional_info.get("is_discontinued", False):
                content_buffer = ""
            try:
                custom_api_key: str | None = user_gpt_context.optional_info.get("api_key")
                async with client.stream(
                    method="POST",
                    url=user_gpt_context.gpt_model.value.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {openai_api_key if custom_api_key is None else custom_api_key}",
                    },  # set headers for openai api request
                    json={
                        "model": user_gpt_context.gpt_model.value.name,
                        "messages": message_history_organizer(user_gpt_context=user_gpt_context),
                        "temperature": user_gpt_context.user_gpt_profile.temperature,
                        "top_p": user_gpt_context.user_gpt_profile.top_p,
                        "n": 1,
                        "stream": True,
                        "presence_penalty": user_gpt_context.user_gpt_profile.presence_penalty,
                        "frequency_penalty": user_gpt_context.user_gpt_profile.frequency_penalty,
                        "max_tokens": min(
                            user_gpt_context.left_tokens, user_gpt_context.gpt_model.value.max_tokens_per_request
                        ),
                        "stop": None,
                        "logit_bias": {},
                        "user": user_gpt_context.user_id,
                    },  # set json for openai api request
                ) as streaming_response:
                    if streaming_response.status_code != 200:  # if status code is not 200
                        err_msg = orjson.loads(await streaming_response.aread()).get("error")
                        if isinstance(err_msg, dict):
                            err_msg = err_msg.get("message")
                        raise GptConnectionException(
                            msg=f"OpenAI Server Error: {err_msg}"
                        )  # raise exception for connection error
                    stream_buffer: str = ""
                    async for stream in streaming_response.aiter_text():  # stream from api
                        stream_buffer += stream
                        for match in ChatGPTConfig.api_regex_pattern.finditer(stream_buffer):  # parse json from stream
                            try:
                                json_data: dict = orjson.loads(match.group(1))["choices"][0]  # data from api
                            except orjson.JSONDecodeError:  # if json is invalid
                                continue
                            finally:
                                stream_buffer = stream_buffer[match.end() :]  # noqa: E203
                            finish_reason: str | None = json_data.get("finish_reason")  # reason for finishing stream
                            delta: str | None = json_data.get("delta")  # generated text from api
                            if finish_reason == "length":
                                raise GptLengthException(
                                    msg="Incomplete model output due to max_tokens parameter or token limit"
                                )  # raise exception for token limit
                            elif finish_reason == "content_filter":
                                raise GptContentFilterException(
                                    msg="Omitted content due to a flag from our content filters"
                                )  # raise exception for openai content filter
                            elif delta is not None:
                                delta_content: str | None = delta.get("content")
                                if delta_content is not None:
                                    content_buffer += delta_content
                                    yield delta_content
            except GptLengthException:
                api_logger.error("token limit exceeded")
                if is_appending_discontinued_message:
                    await MessageManager.set_message_history_safely(
                        user_gpt_context=user_gpt_context,
                        new_content=content_buffer,
                        role=GptRoles.GPT,
                        index=-1,
                    )
                else:
                    await MessageManager.add_message_history_safely(
                        user_gpt_context=user_gpt_context,
                        content=content_buffer,
                        role=GptRoles.GPT,
                    )
                    is_appending_discontinued_message = True
                user_gpt_context.optional_info["is_discontinued"] = True
                continue
            except GptException as gpt_exception:
                api_logger.error(f"gpt exception: {gpt_exception.msg}")
                await MessageManager.rpop_message_history_safely(user_gpt_context=user_gpt_context, role=GptRoles.USER)
                yield gpt_exception.msg
                break
            except httpx.TimeoutException:
                api_logger.error("gpt timeout exception")
                await sleep(ChatGPTConfig.wait_for_reconnect)
                continue
            except Exception as exception:
                api_logger.error(f"unexpected gpt exception: {exception}")
                await MessageManager.rpop_message_history_safely(user_gpt_context=user_gpt_context, role=GptRoles.USER)
                yield "Internal Server Error"
                break
            else:
                await MessageManager.add_message_history_safely(
                    user_gpt_context=user_gpt_context, content=content_buffer, role=GptRoles.GPT
                )
                user_gpt_context.optional_info["is_discontinued"] = False
                break
