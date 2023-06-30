import asyncio
from functools import wraps
from re import Pattern
from typing import Any, AsyncGenerator, Optional
from langchain import PromptTemplate

from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads
import aiohttp
from app.common.config import ChatConfig, config
from app.common.constants import ChatTurnTemplates
from app.errors.chat_exceptions import (
    ChatConnectionException,
    ChatContentFilterException,
    ChatTooMuchTokenException,
    ChatLengthException,
)
from app.models.base_models import MessageHistory
from app.models.llms import ExllamaModel, LlamaCppModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.messages.converter import (
    chat_completion_api_parse_method,
    message_histories_to_list,
    message_histories_to_str,
)
from app.utils.logger import ApiLogger


def _get_stops(*avoids: str) -> list[str]:
    result = []
    for avoid in avoids:
        result.extend(list({avoid, avoid.upper(), avoid.lower(), avoid.capitalize()}))
    return result


def _get_api_key(buffer: BufferedUserContext) -> Optional[str]:
    user_defined_api_key: str | None = (
        buffer.current_user_chat_context.optional_info.get("api_key")
    )
    default_api_key: str | None = getattr(
        buffer.current_llm_model.value, "api_key", None
    )
    return user_defined_api_key if user_defined_api_key is not None else default_api_key


def _get_model_info(buffer: BufferedUserContext) -> tuple[str, str, Pattern]:
    current_model = buffer.current_llm_model.value
    if isinstance(current_model, (LlamaCppModel, ExllamaModel)):
        api_url = config.llama_completion_url
        assert api_url is not None
        model = buffer.current_llm_model.name
        api_regex_pattern = ChatConfig.api_regex_pattern_llama_cpp
    else:
        api_url = current_model.api_url
        model = current_model.name
        api_regex_pattern = ChatConfig.api_regex_pattern_openai
    return api_url, model, api_regex_pattern


async def _handle_stream_error(streaming_response: aiohttp.ClientResponse) -> None:
    if not streaming_response.ok:  # if status code is not 200
        error: Any = orjson_loads(await streaming_response.text()).get("error")
        streaming_response.release()
        ApiLogger("||agenerate_from_chat_completion_api||").error(
            f"API request Error: {error}"
        )
        if isinstance(error, dict):
            error_msg = str(error.get("message"))
            if "maximum context length" in error_msg:
                raise ChatTooMuchTokenException(msg="")
        else:
            error_msg = str(error)
        raise ChatConnectionException(
            msg=f"Chat completion API error: {error_msg}"
        )  # raise exception for connection error


async def _handle_streaming_data(
    streaming_response: aiohttp.ClientResponse,
    json_regex_pattern: Pattern,
    buffer: BufferedUserContext,
    is_chat_completion: bool,
) -> AsyncGenerator[str, None]:
    stream_buffer: bytes = b""
    content_buffer: str = ""
    async for stream, end_of_chunk in streaming_response.content.iter_chunks():  # stream from api
        if buffer.done.is_set():
            raise InterruptedError("Chat was interrupted by the user.")
        stream_buffer += stream
        if not end_of_chunk:
            continue
        for match in json_regex_pattern.finditer(stream_buffer.decode("utf-8")):
            json_data: dict = orjson_loads(match.group(1))
            finish_reason: str | None = json_data["choices"][0]["finish_reason"]
            content: str | None = (
                json_data["choices"][0]["delta"].get("content")
                if is_chat_completion
                else json_data["choices"][0]["text"]
            )
            if finish_reason == "length":
                raise ChatLengthException(
                    msg=content_buffer
                )  # raise exception for token limit
            elif finish_reason == "content_filter":
                raise ChatContentFilterException(
                    msg="Omitted content due to a flag from our content filters"
                )  # raise exception for openai content filter
            if content:
                content_buffer += content
                yield content
            stream_buffer = b""


def _make_data(
    buffer: BufferedUserContext,
    **kwargs: Any,
) -> bytes:
    return orjson_dumps(
        {
            "model": buffer.current_llm_model.name,
            "temperature": buffer.current_user_chat_profile.temperature,
            "top_p": buffer.current_user_chat_profile.top_p,
            "frequency_penalty": buffer.current_user_chat_profile.frequency_penalty,
            "presence_penalty": buffer.current_user_chat_profile.presence_penalty,
            "user": buffer.current_user_chat_context.user_id,
            "n": 1,
            "stream": True,
        }
        | kwargs
    )


def _make_headers(api_key: Optional[str] = None) -> dict[str, str]:
    if api_key is None:
        return {"Content-Type": "application/json"}
    else:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }


def _handle_completion_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ChatTooMuchTokenException as e:
            raise ChatTooMuchTokenException(msg=e.msg)
        except ChatLengthException as e:
            raise ChatLengthException(msg=e.msg)
        except (
            aiohttp.ServerTimeoutError,
            aiohttp.ClientPayloadError,
            asyncio.TimeoutError,
        ):
            pass

    return wrapper


@_handle_completion_error
async def agenerate_from_chat_completion_api(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    api_url, _, api_regex_pattern = _get_model_info(buffer=buffer)

    async with aiohttp.ClientSession(
        timeout=ChatConfig.timeout
    ) as session:  # initialize client
        async with session.post(
            api_url,
            headers=_make_headers(api_key=_get_api_key(buffer=buffer)),
            data=_make_data(
                buffer=buffer,
                max_tokens=max_tokens,
                messages=message_histories_to_list(
                    user_chat_roles=buffer.current_user_chat_roles,
                    parse_method=chat_completion_api_parse_method,
                    user_message_histories=user_message_histories,
                    ai_message_histories=ai_message_histories,
                    system_message_histories=system_message_histories,
                    prefix_prompt=buffer.current_llm_model.value.prefix,
                    prefix_prompt_tokens=buffer.current_llm_model.value.prefix_tokens,
                    suffix_prompt=buffer.current_llm_model.value.suffix,
                    suffix_prompt_tokens=buffer.current_llm_model.value.suffix_tokens,
                ),
            ),  # set json for openai api request
        ) as streaming_response:
            await _handle_stream_error(streaming_response)
            async for token in _handle_streaming_data(
                streaming_response=streaming_response,
                json_regex_pattern=api_regex_pattern,
                buffer=buffer,
                is_chat_completion=True,
            ):
                yield token


@_handle_completion_error
async def agenerate_from_text_completion_api(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    api_url, _, api_regex_pattern = _get_model_info(buffer=buffer)
    chat_turn_prompt: PromptTemplate = (
        buffer.current_llm_model.value.chat_turn_prompt
        if isinstance(buffer.current_llm_model.value, LlamaCppModel)
        else ChatTurnTemplates.ROLE_CONTENT_1
    )

    async with aiohttp.ClientSession(
        timeout=ChatConfig.timeout
    ) as session:  # initialize client
        async with session.post(
            api_url,
            headers=_make_headers(api_key=_get_api_key(buffer=buffer)),
            data=_make_data(
                buffer=buffer,
                max_tokens=max_tokens,
                prompt=message_histories_to_str(
                    user_chat_roles=buffer.current_user_chat_roles,
                    user_message_histories=user_message_histories,
                    ai_message_histories=ai_message_histories,
                    system_message_histories=system_message_histories,
                    prefix_prompt=buffer.current_llm_model.value.prefix,
                    prefix_prompt_tokens=buffer.current_llm_model.value.prefix_tokens,
                    suffix_prompt=buffer.current_llm_model.value.suffix,
                    suffix_prompt_tokens=buffer.current_llm_model.value.suffix_tokens,
                    chat_turn_prompt=chat_turn_prompt,
                ),
                stop=_get_stops(
                    chat_turn_prompt.format(
                        role=buffer.current_llm_model.value.user_chat_roles.user,
                        content="",
                    ).strip(),
                    chat_turn_prompt.format(
                        role=buffer.current_llm_model.value.user_chat_roles.ai,
                        content="",
                    ).strip(),
                ),
                # logit_bias_type="tokens",
                # logit_bias={
                #     b"\u200b".decode("utf-8"): -1,
                # },
            ),  # set json for openai api request
        ) as streaming_response:
            await _handle_stream_error(streaming_response)
            async for token in _handle_streaming_data(
                streaming_response=streaming_response,
                json_regex_pattern=api_regex_pattern,
                buffer=buffer,
                is_chat_completion=False,
            ):
                yield token
