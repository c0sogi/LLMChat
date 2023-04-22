from asyncio import sleep
from itertools import zip_longest
from typing import AsyncGenerator
import json
import httpx

from app.errors.api_exceptions import Responses_500
from app.errors.gpt_exceptions import (
    GptConnectionException,
    GptContentFilterException,
    GptException,
    GptLengthException,
)
from app.utils.chatgpt.chatgpt_message_manager import MessageManager
from app.viewmodels.base_models import SendInitToWebsocket, SendToOpenAI
from app.viewmodels.gpt_models import UserGptContext
from app.utils.chatgpt.chatgpt_config import ChatGPTConfig
from app.utils.logger import logger


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
    if user_gpt_context.is_discontinued:
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
        while True:  # stream until connection is closed
            logger.info("Generating from OpenAI...")
            if not user_gpt_context.is_discontinued:
                gpt_content: str = ""
            try:
                async with client.stream(
                    method="POST",
                    url=user_gpt_context.gpt_model.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {openai_api_key}",
                    },  # set headers for openai api request
                    json={
                        "model": user_gpt_context.gpt_model.name,
                        "messages": message_history_organizer(user_gpt_context=user_gpt_context),
                        "temperature": user_gpt_context.user_gpt_profile.temperature,
                        "top_p": user_gpt_context.user_gpt_profile.top_p,
                        "n": 1,
                        "stream": True,
                        "presence_penalty": user_gpt_context.user_gpt_profile.presence_penalty,
                        "frequency_penalty": user_gpt_context.user_gpt_profile.frequency_penalty,
                        "max_tokens": min(
                            user_gpt_context.left_tokens, user_gpt_context.gpt_model.max_tokens_per_request
                        ),
                        "stop": None,
                        "logit_bias": {},
                        "user": user_gpt_context.user_gpt_profile.user_id,
                    },  # set json for openai api request
                ) as streaming_response:
                    if streaming_response.status_code != 200:  # if status code is not 200
                        error_msg: str = json.loads(await streaming_response.aread()).get("error").get("message")
                        raise GptConnectionException(
                            msg=f"OpenAI 서버로부터 오류: {error_msg}"
                        )  # raise exception for connection error
                    async for raw_text in streaming_response.aiter_text():  # stream from api
                        if raw_text == "data: [DONE]":  # if stream is done
                            break
                        for text in ChatGPTConfig.api_regex_pattern.findall(raw_text):  # parse json from stream
                            try:
                                json_data: dict = json.loads(text)["choices"][0]  # data from api
                            except json.JSONDecodeError:  # if json is invalid
                                continue
                            finish_reason: str | None = json_data.get("finish_reason")  # reason for finishing stream
                            delta: str | None = json_data.get("delta").get("content")  # generated text from api
                            # logger.info(f"finish_reason: {finish_reason}, delta: {delta}")
                            if finish_reason == "length":
                                raise GptLengthException(
                                    msg="Incomplete model output due to max_tokens parameter or token limit"
                                )  # raise exception for token limit
                            elif finish_reason == "content_filter":
                                raise GptContentFilterException(
                                    msg="Omitted content due to a flag from our content filters"
                                )  # raise exception for openai content filter
                            elif delta is not None:
                                gpt_content += delta
                                yield delta
            except GptLengthException:
                logger.error("token limit exceeded")
                if is_appending_discontinued_message:
                    await MessageManager.set_message_history_safely(
                        user_gpt_context=user_gpt_context,
                        new_content=gpt_content,
                        role="gpt",
                        index=-1,
                    )
                else:
                    await MessageManager.add_message_history_safely(
                        user_gpt_context=user_gpt_context,
                        content=gpt_content,
                        role="gpt",
                    )
                    is_appending_discontinued_message = True
                user_gpt_context.is_discontinued = True
                continue
            except GptException as gpt_exception:
                logger.error(f"gpt exception: {gpt_exception.msg}")
                await MessageManager.rpop_message_history_safely(user_gpt_context=user_gpt_context, role="user")
                yield gpt_exception.msg
                break
            except httpx.TimeoutException:
                await sleep(ChatGPTConfig.wait_for_reconnect)
                continue
            except Exception as exception:
                await MessageManager.rpop_message_history_safely(user_gpt_context=user_gpt_context, role="user")
                raise Responses_500.websocket_error(
                    msg=f"unexpected exception generating text from openai: {exception}"
                )
            else:
                await MessageManager.add_message_history_safely(
                    user_gpt_context=user_gpt_context, content=gpt_content, role="gpt"
                )
                user_gpt_context.is_discontinued = False
                break
