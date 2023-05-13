from asyncio import Task, sleep
import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from typing import Any, AsyncGenerator, Union
import orjson
import httpx
from app.errors.gpt_exceptions import (
    GptConnectionException,
    GptContentFilterException,
    GptException,
    GptLengthException,
    GptTextGenerationException,
)
from app.utils.chatgpt.chatgpt_message_manager import MessageManager
from app.viewmodels.base_models import SendInitToWebsocket, SendToStream
from app.models.gpt_models import GptRoles, MessageHistory, UserGptContext
from app.utils.chatgpt.chatgpt_config import ChatGPTConfig
from app.utils.logger import api_logger


def get_tasks_when_end_of_text_generation(
    user_gpt_context: UserGptContext,
    generated_text: str,
    deleted_histories: int,
    calculated_tokens_to_use: int | None = None,
) -> list[Task]:
    tasks: list[Task] = [
        asyncio.create_task(
            MessageManager.add_message_history_safely(
                user_gpt_context=user_gpt_context,
                content=generated_text,
                role=GptRoles.GPT,
                calculated_tokens_to_use=calculated_tokens_to_use,
            )
        )
    ]
    if deleted_histories > 0:
        tasks.extend(
            [
                asyncio.create_task(
                    MessageManager.pop_message_history_safely(
                        user_gpt_context=user_gpt_context,
                        role=role,
                        rpop=False,
                        count=deleted_histories,
                    )
                )
                for role in (GptRoles.GPT, GptRoles.USER)
            ]
        )
    return tasks


def message_history_organizer(
    user_gpt_context: UserGptContext,
    send_to_stream: bool = True,
    return_as_string: bool = False,
) -> Union[list[dict], str]:  # organize message history for openai api
    message_histories: list[dict[str, str]] = []
    if send_to_stream:
        for system_history in user_gpt_context.system_message_histories:
            message_histories.append(SendToStream.from_orm(system_history).dict())  # append system message history
    for user_message_history, gpt_message_history in zip_longest(
        user_gpt_context.user_message_histories,
        user_gpt_context.gpt_message_histories,
    ):
        message_histories.append(
            SendToStream.from_orm(user_message_history).dict()
            if send_to_stream
            else SendInitToWebsocket.from_orm(user_message_history).dict()
        ) if user_message_history is not None else ...  # append user message history
        message_histories.append(
            SendToStream.from_orm(gpt_message_history).dict()
            if send_to_stream
            else SendInitToWebsocket.from_orm(gpt_message_history).dict()
        ) if gpt_message_history is not None else ...  # append gpt message history
    if return_as_string:
        user_role: str = user_gpt_context.user_gpt_profile.user_role
        gpt_role: str = user_gpt_context.user_gpt_profile.gpt_role
        system_role: str = user_gpt_context.user_gpt_profile.system_role
        prefix: str = ""
        if hasattr(user_gpt_context.gpt_model.value, "description"):
            if user_gpt_context.gpt_model.value.description is not None:
                prefix: str = user_gpt_context.gpt_model.value.description.format(
                    user=user_role.upper(),
                    USER=user_role.upper(),
                    gpt=gpt_role.upper(),
                    GPT=gpt_role.upper(),
                    system=system_role.upper(),
                    SYSTEM=system_role.upper(),
                )

        for message_history in message_histories:
            if message_history["role"] == system_role:
                prefix += f"{system_role.upper()}: {message_history['content']}\n"
            elif message_history["role"] == user_role:
                prefix += f"{user_role.upper()}: {message_history['content'].strip()}\n"
            elif message_history["role"] == gpt_role:
                prefix += f"{gpt_role.upper()}: {message_history['content'].strip()}\n"
            else:
                api_logger.error(f"Invalid message history: {message_history}")
                raise Exception("Invalid message history")
        prefix += f"{gpt_role.upper()}: "
        return prefix
    else:
        return message_histories  # return message histories to be used in openai api


async def generate_from_openai(
    user_gpt_context: UserGptContext,  # gpt context for user
) -> AsyncGenerator:  # async generator for streaming
    user_defined_api_key: str | None = user_gpt_context.optional_info.get("api_key")
    default_api_key: str | None = user_gpt_context.gpt_model.value.api_key
    api_key_to_use: Any = user_defined_api_key if user_defined_api_key is not None else default_api_key

    content_buffer: str = ""
    deleted_histories: int = 0
    n_length_exception: int = 0
    async with httpx.AsyncClient(timeout=ChatGPTConfig.wait_for_timeout) as client:  # initialize client
        while True:  # stream until connection is closed
            try:
                messages = message_history_organizer(user_gpt_context=user_gpt_context)
                assert len(messages) > 0 and isinstance(messages, list)
                async with client.stream(
                    method="POST",
                    url=user_gpt_context.gpt_model.value.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key_to_use}",
                    },  # set headers for openai api request
                    json={
                        "model": user_gpt_context.gpt_model.value.name,
                        "messages": messages,
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
                            delta: dict | None = json_data.get("delta")  # generated text from api
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
                if n_length_exception == 0:
                    user_gpt_context.gpt_message_histories.append(
                        MessageHistory(
                            role=user_gpt_context.user_gpt_profile.gpt_role,
                            content=content_buffer + ChatGPTConfig.continue_message,
                            tokens=user_gpt_context.get_tokens_of(content_buffer + ChatGPTConfig.continue_message),
                            is_user=False,
                        )
                    )
                else:
                    old_message: str = user_gpt_context.gpt_message_histories[-1].content.replace(
                        ChatGPTConfig.continue_message, ""
                    )
                    old_tokens: int = user_gpt_context.gpt_message_histories[-1].tokens
                    user_gpt_context.gpt_message_histories[-1].content = old_message + "...[CONTINUED]"
                    user_gpt_context.gpt_message_tokens += (
                        user_gpt_context.get_tokens_of(user_gpt_context.gpt_message_histories[-1].content) - old_tokens
                    )
                deleted_histories += user_gpt_context.ensure_token_not_exceed(
                    extra_token_margin=ChatGPTConfig.extra_token_margin
                )
                n_length_exception += 1
                continue
            except GptException as gpt_exception:
                api_logger.error(f"gpt exception: {gpt_exception.msg}")
                await MessageManager.pop_message_history_safely(user_gpt_context=user_gpt_context, role=GptRoles.USER)
                yield gpt_exception.msg
                break
            except httpx.TimeoutException:
                api_logger.error("gpt timeout exception")
                await sleep(ChatGPTConfig.wait_for_reconnect)
                continue
            except Exception as exception:
                api_logger.error(f"unexpected gpt exception: {exception}")
                await MessageManager.pop_message_history_safely(user_gpt_context=user_gpt_context, role=GptRoles.USER)
                yield "Internal Server Error"
                break
            else:
                await asyncio.gather(
                    *get_tasks_when_end_of_text_generation(
                        user_gpt_context=user_gpt_context,
                        generated_text=content_buffer,
                        deleted_histories=deleted_histories,
                    )
                )
                break


async def generate_from_llama_cpp(
    user_gpt_context: UserGptContext,
    m_queue,
    m_done,
) -> AsyncGenerator:
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    while not m_queue.empty() or not m_done.is_set():
        generation: Any = await loop.run_in_executor(executor, m_queue.get)
        if type(generation) == str:
            yield generation
        elif type(generation) == dict:
            generated_text: str = generation["result"]["generated_text"]
            n_gen_tokens: int = generation["result"]["n_gen_tokens"]
            deleted_histories: int = generation["result"]["deleted_histories"]
            await asyncio.gather(
                *get_tasks_when_end_of_text_generation(
                    user_gpt_context=user_gpt_context,
                    generated_text=generated_text,
                    deleted_histories=deleted_histories,
                    calculated_tokens_to_use=n_gen_tokens,
                )
            )
            break
        else:
            api_logger.error(f"llama_cpp exception: {generation}")
            raise GptTextGenerationException(
                msg="Unexpected response from llama_cpp"
            )  # raise exception for unexpected response


# async def generate_from_llama_cpp(
#     user_gpt_context: UserGptContext,
#     m_queue,
#     m_done,
# ) -> AsyncGenerator:
#     is_appending_discontinued_message: bool = False
#     content_buffer: str = ""
#     loop = asyncio.get_event_loop()
#     executor = ThreadPoolExecutor(max_workers=1)
#     while True:  # stream until connection is closed
#         if not user_gpt_context.optional_info.get("is_discontinued", False):
#             content_buffer = ""
#         try:
#             while not m_queue.empty() or not m_done.is_set():
#                 generation = await loop.run_in_executor(executor, m_queue.get)
#                 if not isinstance(generation, dict):
#                     raise GptTextGenerationException(
#                         msg="Unexpected response from llama_cpp"
#                     )  # raise exception for unexpected response
#                 print(f"generation: {generation}")
#                 finish_reason: str | None = generation["choices"][0]["finish_reason"]
#                 delta: dict | None = generation["choices"][0].get("delta")  # generated text from api
#                 if finish_reason == "length":
#                     raise GptLengthException(
#                         msg="Incomplete model output due to max_tokens parameter or token limit"
#                     )  # raise exception for token limit
#                 elif delta is not None:
#                     delta_content: str | None = delta.get("content")
#                     if delta_content is not None:
#                         content_buffer += delta_content
#                         yield delta_content
#         except GptLengthException:
#             api_logger.error("token limit exceeded")
#             if is_appending_discontinued_message:
#                 await MessageManager.set_message_history_safely(
#                     user_gpt_context=user_gpt_context,
#                     new_content=content_buffer,
#                     role=GptRoles.GPT,
#                     index=-1,
#                 )
#             else:
#                 await MessageManager.add_message_history_safely(
#                     user_gpt_context=user_gpt_context,
#                     content=content_buffer,
#                     role=GptRoles.GPT,
#                 )
#                 is_appending_discontinued_message = True
#             user_gpt_context.optional_info["is_discontinued"] = True
#             continue
#         except GptException as gpt_exception:
#             api_logger.error(f"gpt exception: {gpt_exception.msg}")
#             await MessageManager.rpop_message_history_safely(user_gpt_context=user_gpt_context, role=GptRoles.USER)
#             yield gpt_exception.msg
#             break
#         except Exception as exception:
#             api_logger.error(f"unexpected gpt exception: {exception}", exc_info=True)
#             await MessageManager.rpop_message_history_safely(user_gpt_context=user_gpt_context, role=GptRoles.USER)
#             yield "Internal Server Error"
#             break
#         else:
#             await MessageManager.add_message_history_safely(
#                 user_gpt_context=user_gpt_context, content=content_buffer, role=GptRoles.GPT
#             )
#             user_gpt_context.optional_info["is_discontinued"] = False
#             break
