import asyncio
from typing import Any, AsyncGenerator

from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads
import aiohttp
from app.common.config import ChatConfig, config
from app.common.constants import ChatTurnTemplates
from app.errors.chat_exceptions import (
    ChatConnectionException,
    ChatContentFilterException,
    ChatLengthException,
    ChatTooMuchTokenException,
)
from app.models.base_models import MessageHistory
from app.models.llms import LlamaCppModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.messages.converter import (
    chat_completion_api_parse_method,
    message_histories_to_list,
    message_histories_to_str,
    text_completion_api_parse_method,
)
from app.utils.logger import ApiLogger


def get_stops(s: str) -> list[str]:
    return list({s, s.upper(), s.lower(), s.capitalize()})


async def agenerate_from_chat_completion_api(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    current_model = buffer.current_llm_model.value

    content_buffer: str = ""
    user_defined_api_key: str | None = (
        buffer.current_user_chat_context.optional_info.get("api_key")
    )
    default_api_key: str | None = getattr(current_model, "api_key", None)
    api_key_to_use: str | None = (
        user_defined_api_key if user_defined_api_key is not None else default_api_key
    )

    async with aiohttp.ClientSession(
        timeout=ChatConfig.timeout
    ) as session:  # initialize client
        try:
            if isinstance(current_model, LlamaCppModel):
                api_url = config.llama_cpp_completion_url
                assert api_url is not None
                model = buffer.current_llm_model.name
                api_regex_pattern = ChatConfig.api_regex_pattern_llama_cpp
            else:
                api_url = current_model.api_url
                model = current_model.name
                api_regex_pattern = ChatConfig.api_regex_pattern_openai
            messages = message_histories_to_list(
                user_chat_roles=buffer.current_user_chat_roles,
                parse_method=chat_completion_api_parse_method,
                user_message_histories=user_message_histories,
                ai_message_histories=ai_message_histories,
                system_message_histories=system_message_histories,
                prefix_prompt=current_model.prefix,
                prefix_prompt_tokens=current_model.prefix_tokens,
                suffix_prompt=current_model.suffix,
                suffix_prompt_tokens=current_model.suffix_tokens,
            )

            # ApiLogger("||agenerate_from_chat_completion_api||").debug(
            #     f"Sending these messages to API:\n{messages}"
            # )
            async with session.post(
                api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key_to_use}",
                },  # set headers for openai api request
                data=orjson_dumps(
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": buffer.current_user_chat_profile.temperature,
                        "top_p": buffer.current_user_chat_profile.top_p,
                        "n": 1,
                        "stream": True,
                        "presence_penalty": buffer.current_user_chat_profile.presence_penalty,
                        "frequency_penalty": buffer.current_user_chat_profile.frequency_penalty,
                        "max_tokens": max_tokens,
                        "stop": None,
                        "logit_bias": {},
                        "user": buffer.user_id,
                    }
                ),  # set json for openai api request
            ) as streaming_response:
                if not streaming_response.ok:  # if status code is not 200
                    error: Any = orjson_loads(await streaming_response.text()).get(
                        "error"
                    )
                    streaming_response.release()
                    ApiLogger("||agenerate_from_chat_completion_api||").error(
                        f"API request Error: {error}"
                    )
                    if isinstance(error, dict):
                        error_msg = str(error.get("message"))
                        if "maximum context length" in error_msg:
                            raise ChatTooMuchTokenException(
                                msg=(
                                    f"Maximum context length is {current_model.max_total_tokens} tokens, "
                                    f"however you requested {max_tokens} tokens"
                                )
                            )
                    else:
                        error_msg = str(error)
                    raise ChatConnectionException(
                        msg=f"OpenAI Server Error: {error_msg}"
                    )  # raise exception for connection error
                stream_buffer: bytes = b""
                async for stream, end_of_chunk in streaming_response.content.iter_chunks():  # stream from api
                    if buffer.done.is_set():
                        raise InterruptedError("Chat was interrupted by the user.")
                    stream_buffer += stream
                    if not end_of_chunk:
                        continue
                    for match in api_regex_pattern.finditer(
                        stream_buffer.decode("utf-8")
                    ):
                        json_data: dict = orjson_loads(match.group(1))
                        finish_reason: str | None = json_data["choices"][0][
                            "finish_reason"
                        ]
                        delta_content: str | None = json_data["choices"][0][
                            "delta"
                        ].get("content")
                        if finish_reason == "length":
                            raise ChatLengthException(
                                msg=(
                                    "The token count of your prompt plus max_tokens "
                                    "cannot exceed the model's context length"
                                )
                            )  # raise exception for token limit
                        elif finish_reason == "content_filter":
                            raise ChatContentFilterException(
                                msg="Omitted content due to a flag from our content filters"
                            )  # raise exception for openai content filter
                        if delta_content:
                            content_buffer += delta_content
                            yield delta_content
                        stream_buffer = b""
        except ChatTooMuchTokenException:
            raise ChatTooMuchTokenException(msg=content_buffer)
        except ChatLengthException:
            raise ChatLengthException(msg=content_buffer)
        except (
            aiohttp.ServerTimeoutError,
            aiohttp.ClientPayloadError,
            asyncio.TimeoutError,
        ):
            pass


async def agenerate_from_text_completion_api(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    current_model = buffer.current_llm_model.value

    content_buffer: str = ""
    user_defined_api_key: str | None = (
        buffer.current_user_chat_context.optional_info.get("api_key")
    )
    default_api_key: str | None = getattr(current_model, "api_key", None)
    api_key_to_use: str | None = (
        user_defined_api_key if user_defined_api_key is not None else default_api_key
    )

    async with aiohttp.ClientSession(
        timeout=ChatConfig.timeout
    ) as session:  # initialize client
        try:
            if isinstance(current_model, LlamaCppModel):
                api_url = config.llama_cpp_completion_url
                assert api_url is not None
                model = buffer.current_llm_model.name
                api_regex_pattern = ChatConfig.api_regex_pattern_llama_cpp
                chat_turn_prompt = current_model.chat_turn_prompt
            else:
                api_url = current_model.api_url
                model = current_model.name
                api_regex_pattern = ChatConfig.api_regex_pattern_openai
                chat_turn_prompt = ChatTurnTemplates.ROLE_CONTENT_1
            stop: list[str] = get_stops(
                current_model.user_chat_roles.user + ":",
            ) + get_stops(current_model.user_chat_roles.ai + ":")
            prompt: str = message_histories_to_str(
                user_chat_roles=buffer.current_user_chat_roles,
                user_message_histories=user_message_histories,
                ai_message_histories=ai_message_histories,
                system_message_histories=system_message_histories,
                prefix_prompt=current_model.prefix,
                prefix_prompt_tokens=current_model.prefix_tokens,
                suffix_prompt=current_model.suffix,
                suffix_prompt_tokens=current_model.suffix_tokens,
                chat_turn_prompt=chat_turn_prompt,
            )

            # ApiLogger("||agenerate_from_text_completion_api||").debug(
            #     f"Sending these prompt to API:\n{prompt}"
            # )
            async with session.post(
                api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key_to_use}",
                },  # set headers for openai api request
                data=orjson_dumps(
                    {
                        "model": model,
                        "prompt": prompt,
                        "temperature": buffer.current_user_chat_profile.temperature,
                        "top_p": buffer.current_user_chat_profile.top_p,
                        "n": 1,
                        "stream": True,
                        "presence_penalty": buffer.current_user_chat_profile.presence_penalty,
                        "frequency_penalty": buffer.current_user_chat_profile.frequency_penalty,
                        "max_tokens": max_tokens,
                        "stop": stop,
                        "logit_bias": {},
                        "user": buffer.user_id,
                    }
                ),  # set json for openai api request
            ) as streaming_response:
                if not streaming_response.ok:  # if status code is not 200
                    error: Any = orjson_loads(await streaming_response.text()).get(
                        "error"
                    )
                    streaming_response.release()
                    ApiLogger("||agenerate_from_text_completion_api||").error(
                        f"API request Error: {error}"
                    )
                    if isinstance(error, dict):
                        error_msg = str(error.get("message"))
                        if "maximum context length" in error_msg:
                            raise ChatTooMuchTokenException(
                                msg=(
                                    f"Maximum context length is {current_model.max_total_tokens} tokens, "
                                    f"however you requested {max_tokens} tokens"
                                )
                            )
                    else:
                        error_msg = str(error)
                    raise ChatConnectionException(
                        msg=f"OpenAI Server Error: {error_msg}"
                    )  # raise exception for connection error
                stream_buffer: bytes = b""
                async for stream, end_of_chunk in streaming_response.content.iter_chunks():  # stream from api
                    if buffer.done.is_set():
                        raise InterruptedError("Chat was interrupted by the user.")
                    stream_buffer += stream
                    if not end_of_chunk:
                        continue
                    for match in api_regex_pattern.finditer(
                        stream_buffer.decode("utf-8")
                    ):
                        json_data: dict = orjson_loads(match.group(1))
                        finish_reason: str | None = json_data["choices"][0][
                            "finish_reason"
                        ]
                        text: str | None = json_data["choices"][0].get("text")
                        if finish_reason == "length":
                            raise ChatLengthException(
                                msg=(
                                    "The token count of your prompt plus max_tokens "
                                    "cannot exceed the model's context length"
                                )
                            )  # raise exception for token limit
                        elif finish_reason == "content_filter":
                            raise ChatContentFilterException(
                                msg="Omitted content due to a flag from our content filters"
                            )  # raise exception for openai content filter
                        if text:
                            content_buffer += text
                            yield text
                        stream_buffer = b""
        except ChatTooMuchTokenException:
            raise ChatTooMuchTokenException(msg=content_buffer)
        except ChatLengthException:
            raise ChatLengthException(msg=content_buffer)
        except (
            aiohttp.ServerTimeoutError,
            aiohttp.ClientPayloadError,
            asyncio.TimeoutError,
        ):
            pass
