from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Any, AsyncGenerator, Generator, Optional

import aiohttp
import openai
from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads

from app.shared import Shared
from app.errors.chat_exceptions import (
    ChatConnectionException,
    ChatContentFilterException,
    ChatLengthException,
    ChatTextGenerationException,
)
from app.models.llms import LlamaCppModel, OpenAIModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.chat_config import ChatConfig
from app.utils.chat.llama_cpp import llama_cpp_generation
from app.utils.chat.prompts import message_history_organizer
from app.utils.logger import api_logger


def generate_from_llama_cpp(buffer: BufferedUserContext) -> Generator:
    llama_cpp_model = buffer.current_llm_model.value
    assert isinstance(llama_cpp_model, LlamaCppModel), type(llama_cpp_model)

    shared: Shared = Shared()
    m_queue = shared.process_manager.Queue()
    m_done = shared.process_manager.Event()
    process_pool_executor: ProcessPoolExecutor = shared.process_pool_executor
    try:
        future_exception: Optional[BaseException] = None
        future: Future[None] = process_pool_executor.submit(
            llama_cpp_generation,
            user_chat_context=buffer.current_user_chat_context,
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
                api_logger.exception("An error occurred during llama_cpp_generation.")
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


async def agenerate_from_openai(buffer: BufferedUserContext) -> AsyncGenerator[str, None]:
    current_model = buffer.current_llm_model.value
    messages = message_history_organizer(user_chat_context=buffer.current_user_chat_context)
    assert isinstance(messages, list)
    assert isinstance(current_model, OpenAIModel)

    content_buffer: str = ""
    user_defined_api_key: str | None = buffer.current_user_chat_context.optional_info.get("api_key")
    default_api_key: str | None = current_model.api_key
    api_key_to_use: Any = user_defined_api_key if user_defined_api_key is not None else default_api_key

    async with aiohttp.ClientSession() as session:  # initialize client
        try:
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
                        "max_tokens": min(
                            buffer.current_user_chat_context.left_tokens,
                            current_model.max_tokens_per_request,
                        ),
                        "stop": None,
                        "logit_bias": {},
                        "user": buffer.user_id,
                    }
                ),  # set json for openai api request
                timeout=aiohttp.ClientTimeout(total=ChatConfig.wait_for_timeout),
            ) as streaming_response:
                if streaming_response.status != 200:  # if status code is not 200
                    err_msg = orjson_loads(await streaming_response.text()).get("error")
                    if isinstance(err_msg, dict):
                        err_msg = err_msg.get("message")
                    raise ChatConnectionException(
                        msg=f"OpenAI Server Error: {err_msg}"
                    )  # raise exception for connection error
                stream_buffer: bytes = b""
                async for stream, end_of_chunk in streaming_response.content.iter_chunks():  # stream from api
                    stream_buffer += stream
                    if not end_of_chunk:
                        continue
                    for match in ChatConfig.api_regex_pattern.finditer(stream_buffer.decode("utf-8")):
                        json_data: dict = orjson_loads(match.group(1))
                        finish_reason: str | None = json_data["choices"][0]["finish_reason"]
                        delta_content: str | None = json_data["choices"][0]["delta"].get("content")
                        if finish_reason == "length":
                            raise ChatLengthException(
                                msg="Incomplete model output due to max_tokens parameter or token limit"
                            )  # raise exception for token limit
                        elif finish_reason == "content_filter":
                            raise ChatContentFilterException(
                                msg="Omitted content due to a flag from our content filters"
                            )  # raise exception for openai content filter
                        if isinstance(delta_content, str):
                            content_buffer += delta_content
                            yield delta_content
                    stream_buffer = b""
        except ChatLengthException:
            raise ChatLengthException(msg=content_buffer)
        except (aiohttp.ServerTimeoutError, aiohttp.ClientPayloadError):
            pass


def generate_from_openai(buffer: BufferedUserContext) -> Generator[str, None, None]:
    assert isinstance(buffer.current_user_chat_context.llm_model.value, OpenAIModel)
    user_defined_api_key: str | None = buffer.current_user_chat_context.optional_info.get("api_key")
    default_api_key: str | None = buffer.current_user_chat_context.llm_model.value.api_key
    api_key_to_use: Any = user_defined_api_key if user_defined_api_key is not None else default_api_key

    current_model = buffer.current_llm_model.value
    content_buffer: str = ""
    messages = message_history_organizer(user_chat_context=buffer.current_user_chat_context)
    assert isinstance(messages, list)
    assert isinstance(current_model, OpenAIModel)
    try:
        for stream in openai.ChatCompletion.create(
            api_key=api_key_to_use,
            model=current_model.name,
            messages=messages,
            temperature=buffer.current_user_chat_profile.temperature,
            top_p=buffer.current_user_chat_profile.top_p,
            n=1,
            stream=True,
            presence_penalty=buffer.current_user_chat_profile.presence_penalty,
            frequency_penalty=buffer.current_user_chat_profile.frequency_penalty,
            max_tokens=min(
                buffer.current_user_chat_context.left_tokens,
                buffer.current_user_chat_context.llm_model.value.max_tokens_per_request,
            ),
            stop=None,
            logit_bias={},
            user=buffer.current_user_chat_context.user_id,
        ):  # stream from api
            assert isinstance(stream, dict)
            finish_reason: str | None = stream["choices"][0]["finish_reason"]
            delta_content: str | None = stream["choices"][0]["delta"].get("content")
            if finish_reason == "length":
                raise ChatLengthException(
                    msg="Incomplete model output due to max_tokens parameter or token limit"
                )  # raise exception for token limit
            elif finish_reason == "content_filter":
                raise ChatContentFilterException(
                    msg="Omitted content due to a flag from our content filters"
                )  # raise exception for openai content filter
            if delta_content is not None:
                content_buffer += delta_content
                yield delta_content
    except ChatLengthException:
        raise ChatLengthException(msg=content_buffer)
