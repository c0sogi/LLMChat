from typing import Any, AsyncGenerator

import asyncio
from langchain.callbacks import AsyncIteratorCallbackHandler
from app.errors.chat_exceptions import ChatTextGenerationException
from openai.error import OpenAIError
from app.models.chat_models import MessageHistory

from app.models.llms import OpenAIModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.messages.converter import (
    message_histories_to_list,
    openai_parse_method,
)
from app.utils.langchain.chat_openai import CustomChatOpenAI


async def agenerate_from_openai(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    current_model = buffer.current_llm_model.value
    assert isinstance(current_model, OpenAIModel)

    user_defined_api_key: str | None = (
        buffer.current_user_chat_context.optional_info.get("api_key")
    )
    default_api_key: str | None = current_model.api_key
    api_key_to_use: Any = (
        user_defined_api_key if user_defined_api_key is not None else default_api_key
    )

    callback = AsyncIteratorCallbackHandler()
    producer = CustomChatOpenAI(
        client=None,
        model_name=current_model.name,  # type: ignore
        openai_api_key=api_key_to_use,
        streaming=True,
        max_tokens=max_tokens,
        temperature=buffer.current_user_chat_profile.temperature,
        model_kwargs={
            "top_p": buffer.current_user_chat_profile.top_p,
            "presence_penalty": buffer.current_user_chat_profile.presence_penalty,
            "frequency_penalty": buffer.current_user_chat_profile.frequency_penalty,
            "logit_bias": {},
            "user": buffer.user_id,
        },
    ).agenerate(
        stop=None,
        messages=[
            message_histories_to_list(
                user_chat_roles=buffer.current_user_chat_roles,
                parse_method=openai_parse_method,
                user_message_histories=user_message_histories,
                ai_message_histories=ai_message_histories,
                system_message_histories=system_message_histories,
                prefix_prompt=current_model.prefix,
                prefix_prompt_tokens=current_model.prefix_tokens,
                suffix_prompt=current_model.suffix,
                suffix_prompt_tokens=current_model.suffix_tokens,
            ),
        ],
        callbacks=[callback],
    )

    producer_task = asyncio.create_task(producer)

    try:
        async for token in callback.aiter():
            if buffer.done.is_set():
                raise InterruptedError("Chat was interrupted by the user.")
            yield token
        await producer_task

    except OpenAIError as e:
        raise ChatTextGenerationException(msg=str(e))

    finally:
        if not producer_task.done():
            producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass


# async def agenerate_from_openai(
#     buffer: BufferedUserContext,
#     user_message_histories: list[MessageHistory],
#     ai_message_histories: list[MessageHistory],
#     system_message_histories: list[MessageHistory],
#     max_tokens: int,
# ) -> AsyncGenerator[str, None]:
#     current_model = buffer.current_llm_model.value
#     assert isinstance(current_model, OpenAIModel)

#     content_buffer: str = ""
#     user_defined_api_key: str | None = (
#         buffer.current_user_chat_context.optional_info.get("api_key")
#     )
#     default_api_key: str | None = current_model.api_key
#     api_key_to_use: Any = (
#         user_defined_api_key if user_defined_api_key is not None else default_api_key
#     )

#     async with aiohttp.ClientSession(
#         timeout=chat_config.timeout
#     ) as session:  # initialize client
#         try:
#             messages = message_histories_to_list(
#                 user_chat_roles=buffer.current_user_chat_roles,
#                 parse_method=openai_parse_method,
#                 user_message_histories=user_message_histories,
#                 ai_message_histories=ai_message_histories,
#                 system_message_histories=system_message_histories,
#                 prefix_prompt=current_model.prefix,
#                 prefix_prompt_tokens=current_model.prefix_tokens,
#                 suffix_prompt=current_model.suffix,
#                 suffix_prompt_tokens=current_model.suffix_tokens,
#             )
#             ApiLogger("|A01|").debug(f"Sending these messages to OpenAI:\n{messages}")
#             async with session.post(
#                 current_model.api_url,
#                 headers={
#                     "Content-Type": "application/json",
#                     "Authorization": f"Bearer {api_key_to_use}",
#                 },  # set headers for openai api request
#                 data=orjson_dumps(
#                     {
#                         "model": current_model.name,
#                         "messages": messages,
#                         "temperature": buffer.current_user_chat_profile.temperature,
#                         "top_p": buffer.current_user_chat_profile.top_p,
#                         "n": 1,
#                         "stream": True,
#                         "presence_penalty": buffer.current_user_chat_profile.presence_penalty,
#                         "frequency_penalty": buffer.current_user_chat_profile.frequency_penalty,
#                         "max_tokens": max_tokens,
#                         "stop": None,
#                         "logit_bias": {},
#                         "user": buffer.user_id,
#                     }
#                 ),  # set json for openai api request
#             ) as streaming_response:
#                 if not streaming_response.ok:  # if status code is not 200
#                     error: Any = orjson_loads(await streaming_response.text()).get(
#                         "error"
#                     )
#                     streaming_response.release()
#                     api_logger.error(f"OpenAI Server Error: {error}")
#                     if isinstance(error, dict):
#                         error_msg = str(error.get("message"))
#                         if "maximum context length" in error_msg:
#                             raise ChatTooMuchTokenException(
#                                 msg=(
#                                     f"Maximum context length is {current_model.max_total_tokens} tokens, "
#                                     f"however you requested {max_tokens} tokens"
#                                 )
#                             )
#                     else:
#                         error_msg = str(error)
#                     raise ChatConnectionException(
#                         msg=f"OpenAI Server Error: {error_msg}"
#                     )  # raise exception for connection error
#                 stream_buffer: bytes = b""
#                 async for stream, end_of_chunk in streaming_response.content.iter_chunks():  # stream from api
#                     stream_buffer += stream
#                     if not end_of_chunk:
#                         continue
#                     for match in chat_config.api_regex_pattern.finditer(
#                         stream_buffer.decode("utf-8")
#                     ):
#                         json_data: dict = orjson_loads(match.group(1))
#                         finish_reason: str | None = json_data["choices"][0][
#                             "finish_reason"
#                         ]
#                         delta_content: str | None = json_data["choices"][0][
#                             "delta"
#                         ].get("content")
#                         if finish_reason == "length":
#                             raise ChatLengthException(
#                                 msg=(
#                                     "The token count of your prompt plus max_tokens "
#                                     "cannot exceed the model's context length"
#                                 )
#                             )  # raise exception for token limit
#                         elif finish_reason == "content_filter":
#                             raise ChatContentFilterException(
#                                 msg="Omitted content due to a flag from our content filters"
#                             )  # raise exception for openai content filter
#                         if isinstance(delta_content, str):
#                             content_buffer += delta_content
#                             yield delta_content
#                     stream_buffer = b""
#         except ChatTooMuchTokenException:
#             raise ChatTooMuchTokenException(msg=content_buffer)
#         except ChatLengthException:
#             raise ChatLengthException(msg=content_buffer)
#         except (
#             aiohttp.ServerTimeoutError,
#             aiohttp.ClientPayloadError,
#             asyncio.TimeoutError,
#         ):
#             pass
