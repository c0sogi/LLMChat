from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from typing import Any, AsyncGenerator, Generator, Optional

import aiohttp
import asyncio
from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads
from app.common.config import ChatConfig, chat_config
from app.models.chat_models import MessageHistory

from app.shared import Shared
from app.errors.chat_exceptions import (
    ChatConnectionException,
    ChatContentFilterException,
    ChatLengthException,
    ChatTextGenerationException,
    ChatTooMuchTokenException,
)
from app.models.llms import LlamaCppModel, OpenAIModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.llama_cpp import llama_cpp_generation
from app.utils.chat.prompts import (
    message_histories_to_list,
    message_histories_to_str,
    openai_parse_method,
)
from app.utils.logger import ApiLogger, api_logger


def generate_from_llama_cpp(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> Generator:
    llama_cpp_model = buffer.current_llm_model.value
    assert isinstance(llama_cpp_model, LlamaCppModel), type(llama_cpp_model)

    shared: Shared = Shared()
    m_queue = shared.process_manager.Queue()
    m_done = shared.process_manager.Event()
    process_pool_executor: ProcessPoolExecutor = shared.process_pool_executor
    try:
        prompt = message_histories_to_str(
            user_chat_roles=buffer.current_user_chat_roles,
            user_message_histories=user_message_histories,
            ai_message_histories=ai_message_histories,
            system_message_histories=system_message_histories,
            prefix_prompt=llama_cpp_model.prefix,
            prefix_prompt_tokens=llama_cpp_model.prefix_tokens,
            suffix_prompt=llama_cpp_model.suffix,
            suffix_prompt_tokens=llama_cpp_model.suffix_tokens,
            chat_turn_prompt=llama_cpp_model.chat_turn_prompt,
        )
        api_logger.info(f"Sending this prompt to llama_cpp:\n{prompt}")
        future_exception: Optional[BaseException] = None
        future: Future[None] = process_pool_executor.submit(
            llama_cpp_generation,
            user_chat_context=buffer.current_user_chat_context,
            prompt=prompt,
            max_tokens=max_tokens,
            m_queue=m_queue,
            m_done=m_done,
        )
        while not future.done():
            if buffer.done.is_set():
                m_done.set()
                # wait for the future to complete for a certain amount of time
                _, not_done = wait([future], return_when=FIRST_COMPLETED)

                # if future is still not done, try cancelling
                if future in not_done:
                    future.cancel()
                break
            generation: Any = m_queue.get()
            if isinstance(generation, str):
                yield generation
            elif isinstance(generation, BaseException):
                api_logger.exception(
                    f"An error occurred during llama_cpp_generation: {generation} {type(generation)}"
                )
                future_exception = generation
            else:
                break
        if future_exception is not None:
            raise ChatTextGenerationException(
                msg="Unexpected response from llama_cpp"
            )  # raise exception for unexpected response
    except BrokenProcessPool as e:
        process_pool_executor.shutdown(wait=False)
        shared.process_pool_executor = ProcessPoolExecutor()
        api_logger.exception(f"BrokenProcessPool: {e}")
        raise ChatConnectionException(msg="BrokenProcessPool")


async def agenerate_from_openai(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    current_model = buffer.current_llm_model.value
    assert isinstance(current_model, OpenAIModel)

    content_buffer: str = ""
    user_defined_api_key: str | None = (
        buffer.current_user_chat_context.optional_info.get("api_key")
    )
    default_api_key: str | None = current_model.api_key
    api_key_to_use: Any = (
        user_defined_api_key if user_defined_api_key is not None else default_api_key
    )

    async with aiohttp.ClientSession(
        timeout=chat_config.timeout
    ) as session:  # initialize client
        try:
            messages = message_histories_to_list(
                user_chat_roles=buffer.current_user_chat_roles,
                parse_method=openai_parse_method,
                user_message_histories=user_message_histories,
                ai_message_histories=ai_message_histories,
                system_message_histories=system_message_histories,
                prefix_prompt=current_model.prefix,
                prefix_prompt_tokens=current_model.prefix_tokens,
                suffix_prompt=current_model.suffix,
                suffix_prompt_tokens=current_model.suffix_tokens,
            )
            ApiLogger("|A01|").debug(f"Sending these messages to OpenAI:\n{messages}")
            async with session.post(
                current_model.api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key_to_use}",
                },  # set headers for openai api request
                data=orjson_dumps(
                    {
                        "model": current_model.name,
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
                    streaming_response.release()
                    error: Any = orjson_loads(await streaming_response.text()).get(
                        "error"
                    )
                    api_logger.error(f"OpenAI Server Error: {error}")
                    if isinstance(error, dict):
                        error_msg = str(error.get("message"))
                        if "maximum context length" in error_msg:
                            raise ChatTooMuchTokenException(
                                msg=(
                                    f"This model’s maximum context length is {current_model.max_total_tokens} tokens, "
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
                    stream_buffer += stream
                    if not end_of_chunk:
                        continue
                    for match in chat_config.api_regex_pattern.finditer(
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
                        if isinstance(delta_content, str):
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


# def generate_from_openai(
#     buffer: BufferedUserContext,
#     user_message_histories: list[MessageHistory],
#     ai_message_histories: list[MessageHistory],
#     system_message_histories: list[MessageHistory],
#     max_tokens: int,
# ) -> Generator[str, None, None]:
#     def parse_method(message_history: MessageHistory) -> dict[str, str]:
#         return OpenAIChatMessage.from_orm(message_history).dict()

#     current_model = buffer.current_llm_model.value
#     assert isinstance(current_model, OpenAIModel)

#     content_buffer: str = ""
#     user_defined_api_key: str | None = buffer.current_user_chat_context.optional_info.get("api_key")
#     default_api_key: str | None = current_model.api_key
#     api_key_to_use: Any = user_defined_api_key if user_defined_api_key is not None else default_api_key

#     try:
#         for stream in openai.ChatCompletion.create(
#             api_key=api_key_to_use,
#             model=current_model.name,
#             messages=message_histories_to_list(
#                 parse_method=parse_method,
#                 user_message_histories=buffer.current_user_message_histories,
#                 ai_message_histories=buffer.current_ai_message_histories,
#                 system_message_histories=buffer.current_system_message_histories,
#             ),
#             temperature=buffer.current_user_chat_profile.temperature,
#             top_p=buffer.current_user_chat_profile.top_p,
#             n=1,
#             stream=True,
#             presence_penalty=buffer.current_user_chat_profile.presence_penalty,
#             frequency_penalty=buffer.current_user_chat_profile.frequency_penalty,
#             max_tokens=max_tokens,
#             stop=None,
#             logit_bias={},
#             user=buffer.current_user_chat_context.user_id,
#         ):  # stream from api
#             assert isinstance(stream, dict)
#             finish_reason: str | None = stream["choices"][0]["finish_reason"]
#             delta_content: str | None = stream["choices"][0]["delta"].get("content")
#             if finish_reason == "length":
#                 raise ChatLengthException(
#                     msg="Incomplete model output due to max_tokens parameter or token limit"
#                 )  # raise exception for token limit
#             elif finish_reason == "content_filter":
#                 raise ChatContentFilterException(
#                     msg="Omitted content due to a flag from our content filters"
#                 )  # raise exception for openai content filter
#             if delta_content is not None:
#                 content_buffer += delta_content
#                 yield delta_content
#     except ChatLengthException:
#         raise ChatLengthException(msg=content_buffer)


async def get_summarization(
    to_summarize: str,
    to_summarize_tokens: Optional[int] = None,
) -> str:
    shared = Shared()
    if to_summarize_tokens is None:
        to_summarize_tokens = len(
            shared.token_text_splitter._tokenizer.encode(to_summarize)
        )

    if to_summarize_tokens < ChatConfig.summarization_token_limit:
        summarize_chain = shared.stuff_summarize_chain
    else:
        summarize_chain = shared.map_reduce_summarize_chain
    result = await summarize_chain.arun(
        shared.token_text_splitter.create_documents([to_summarize])
    )
    api_logger.info(f"Summarization result:\n {result}")
    return result
