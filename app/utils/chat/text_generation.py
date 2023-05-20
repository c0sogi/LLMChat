from asyncio import sleep
import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from typing import Any, AsyncGenerator, Coroutine, Union
import orjson
import httpx
from app.errors.chat_exceptions import (
    ChatConnectionException,
    ChatContentFilterException,
    ChatLengthException,
    ChatTextGenerationException,
)
from app.models.llms import OpenAIModel
from app.utils.chat.message_manager import MessageManager
from app.utils.chat.prompts import ROLE_CONTENT_TMPL_CHAT1
from app.viewmodels.base_models import SendInitToWebsocket, SendToStream
from app.models.chat_models import ChatRoles, UserChatContext
from app.utils.chat.chat_config import ChatConfig
from app.utils.logger import api_logger


def get_coroutines_when_end_of_text_generation(
    user_chat_context: UserChatContext,
    generated_text: str,
    deleted_histories: int,
    calculated_tokens_to_use: int | None = None,
) -> list[Coroutine]:
    coroutines: list[Coroutine] = [
        MessageManager.add_message_history_safely(
            user_chat_context=user_chat_context,
            content=generated_text,
            role=ChatRoles.AI,
            calculated_tokens_to_use=calculated_tokens_to_use,
        )
    ]
    if deleted_histories > 0:
        coroutines.extend(
            [
                MessageManager.pop_message_history_safely(
                    user_chat_context=user_chat_context,
                    role=role,
                    rpop=False,
                    count=deleted_histories,
                )
                for role in (ChatRoles.AI, ChatRoles.USER)
            ]
        )
    return coroutines


def message_history_organizer(
    user_chat_context: UserChatContext,
    send_to_stream: bool = True,
    return_as_string: bool = False,
    chat_turn_prompt: str = ROLE_CONTENT_TMPL_CHAT1,
) -> Union[list[dict], str]:  # organize message history for openai api
    message_histories: list[dict[str, str]] = []
    if send_to_stream:
        for system_history in user_chat_context.system_message_histories:
            message_histories.append(SendToStream.from_orm(system_history).dict())  # append system message history
    for user_message_history, ai_message_history in zip_longest(
        user_chat_context.user_message_histories,
        user_chat_context.ai_message_histories,
    ):
        message_histories.append(
            SendToStream.from_orm(user_message_history).dict()
            if send_to_stream
            else SendInitToWebsocket.from_orm(user_message_history).dict()
        ) if user_message_history is not None else ...  # append user message history
        message_histories.append(
            SendToStream.from_orm(ai_message_history).dict()
            if send_to_stream
            else SendInitToWebsocket.from_orm(ai_message_history).dict()
        ) if ai_message_history is not None else ...  # append gpt message history
    if return_as_string:
        user_role: str = user_chat_context.user_chat_profile.user_role
        ai_role: str = user_chat_context.user_chat_profile.ai_role
        system_role: str = user_chat_context.user_chat_profile.system_role
        prefix: str = ""
        if hasattr(user_chat_context.llm_model.value, "description"):
            if user_chat_context.llm_model.value.description is not None:  # type: ignore
                prefix: str = user_chat_context.llm_model.value.description.format(  # type: ignore
                    user=user_role.upper(),
                    USER=user_role.upper(),
                    ai=ai_role.upper(),
                    AI=ai_role.upper(),
                    system=system_role.upper(),
                    SYSTEM=system_role.upper(),
                )

        for message_history in message_histories:
            if message_history["role"] == system_role:
                prefix += chat_turn_prompt.format(role=system_role.upper(), content=message_history["content"].strip())
            elif message_history["role"] == user_role:
                prefix += chat_turn_prompt.format(role=user_role.upper(), content=message_history["content"].strip())
            elif message_history["role"] == ai_role:
                prefix += chat_turn_prompt.format(role=ai_role.upper(), content=message_history["content"].strip())
            else:
                api_logger.error(f"Invalid message history: {message_history}")
                raise Exception("Invalid message history")
        prefix += chat_turn_prompt.format(role=ai_role.upper(), content="").strip() + " "
        return prefix
    else:
        return message_histories  # return message histories to be used in openai api


async def generate_from_openai(user_chat_context: UserChatContext) -> AsyncGenerator:  # async generator for streaming
    assert isinstance(user_chat_context.llm_model.value, OpenAIModel)
    user_defined_api_key: str | None = user_chat_context.optional_info.get("api_key")
    default_api_key: str | None = user_chat_context.llm_model.value.api_key
    api_key_to_use: Any = user_defined_api_key if user_defined_api_key is not None else default_api_key

    content_buffer: str = ""
    deleted_histories: int = 0
    async with httpx.AsyncClient(timeout=ChatConfig.wait_for_timeout) as client:  # initialize client
        while True:  # stream until connection is closed
            try:
                messages = message_history_organizer(user_chat_context=user_chat_context)
                assert len(messages) > 0 and isinstance(messages, list)
                async with client.stream(
                    method="POST",
                    url=user_chat_context.llm_model.value.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key_to_use}",
                    },  # set headers for openai api request
                    json={
                        "model": user_chat_context.llm_model.value.name,
                        "messages": messages,
                        "temperature": user_chat_context.user_chat_profile.temperature,
                        "top_p": user_chat_context.user_chat_profile.top_p,
                        "n": 1,
                        "stream": True,
                        "presence_penalty": user_chat_context.user_chat_profile.presence_penalty,
                        "frequency_penalty": user_chat_context.user_chat_profile.frequency_penalty,
                        "max_tokens": min(
                            user_chat_context.left_tokens, user_chat_context.llm_model.value.max_tokens_per_request
                        ),
                        "stop": None,
                        "logit_bias": {},
                        "user": user_chat_context.user_id,
                    },  # set json for openai api request
                ) as streaming_response:
                    if streaming_response.status_code != 200:  # if status code is not 200
                        err_msg = orjson.loads(await streaming_response.aread()).get("error")
                        if isinstance(err_msg, dict):
                            err_msg = err_msg.get("message")
                        raise ChatConnectionException(
                            msg=f"OpenAI Server Error: {err_msg}"
                        )  # raise exception for connection error
                    stream_buffer: str = ""
                    async for stream in streaming_response.aiter_text():  # stream from api
                        stream_buffer += stream
                        for match in ChatConfig.api_regex_pattern.finditer(stream_buffer):  # parse json from stream
                            try:
                                json_data: dict = orjson.loads(match.group(1))["choices"][0]  # data from api
                            except orjson.JSONDecodeError:  # if json is invalid
                                continue
                            finally:
                                stream_buffer = stream_buffer[match.end() :]  # noqa: E203
                            finish_reason: str | None = json_data.get("finish_reason")  # reason for finishing stream
                            delta: dict | None = json_data.get("delta")  # generated text from api
                            if finish_reason == "length":
                                raise ChatLengthException(
                                    msg="Incomplete model output due to max_tokens parameter or token limit"
                                )  # raise exception for token limit
                            elif finish_reason == "content_filter":
                                raise ChatContentFilterException(
                                    msg="Omitted content due to a flag from our content filters"
                                )  # raise exception for openai content filter
                            elif delta is not None:
                                delta_content: str | None = delta.get("content")
                                if delta_content is not None:
                                    content_buffer += delta_content
                                    yield delta_content
            except ChatLengthException:
                api_logger.error("token limit exceeded")
                if len(user_chat_context.user_message_histories) == len(user_chat_context.ai_message_histories):
                    deleted_while_overriding: int | None = await MessageManager.set_message_history_safely(
                        user_chat_context=user_chat_context,
                        role=ChatRoles.AI,
                        index=-1,
                        new_content=content_buffer.replace(ChatConfig.continue_message, "")
                        + ChatConfig.continue_message,
                        update_cache=False,
                        extra_token_margin=ChatConfig.extra_token_margin,
                    )
                    if deleted_while_overriding is not None:
                        deleted_histories += deleted_while_overriding

                else:
                    deleted_histories = await MessageManager.add_message_history_safely(
                        user_chat_context=user_chat_context,
                        role=ChatRoles.AI,
                        content=content_buffer,
                        update_cache=False,
                        extra_token_margin=ChatConfig.extra_token_margin,
                    )
                continue
            except httpx.TimeoutException:
                api_logger.error("openai timeout exception")
                await sleep(ChatConfig.wait_for_reconnect)
                continue
            else:
                await asyncio.gather(
                    *get_coroutines_when_end_of_text_generation(
                        user_chat_context=user_chat_context,
                        generated_text=content_buffer,
                        deleted_histories=deleted_histories,
                    )
                )
                break


async def generate_from_llama_cpp(
    user_chat_context: UserChatContext,
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
                *get_coroutines_when_end_of_text_generation(
                    user_chat_context=user_chat_context,
                    generated_text=generated_text,
                    deleted_histories=deleted_histories,
                    calculated_tokens_to_use=n_gen_tokens,
                )
            )
            break
        else:
            api_logger.error(f"llama_cpp exception: {type(generation)}{str(generation)}")
            raise ChatTextGenerationException(
                msg="Unexpected response from llama_cpp"
            )  # raise exception for unexpected response


# async def generate_from_llama_cpp(
#     user_chat_context: UserChatContext,
#     m_queue,
#     m_done,
# ) -> AsyncGenerator:
#     is_appending_discontinued_message: bool = False
#     content_buffer: str = ""
#     loop = asyncio.get_event_loop()
#     executor = ThreadPoolExecutor(max_workers=1)
#     while True:  # stream until connection is closed
#         if not user_chat_context.optional_info.get("is_discontinued", False):
#             content_buffer = ""
#         try:
#             while not m_queue.empty() or not m_done.is_set():
#                 generation = await loop.run_in_executor(executor, m_queue.get)
#                 if not isinstance(generation, dict):
#                     raise ChatTextGenerationException(
#                         msg="Unexpected response from llama_cpp"
#                     )  # raise exception for unexpected response
#                 print(f"generation: {generation}")
#                 finish_reason: str | None = generation["choices"][0]["finish_reason"]
#                 delta: dict | None = generation["choices"][0].get("delta")  # generated text from api
#                 if finish_reason == "length":
#                     raise ChatLengthException(
#                         msg="Incomplete model output due to max_tokens parameter or token limit"
#                     )  # raise exception for token limit
#                 elif delta is not None:
#                     delta_content: str | None = delta.get("content")
#                     if delta_content is not None:
#                         content_buffer += delta_content
#                         yield delta_content
#         except ChatLengthException:
#             api_logger.error("token limit exceeded")
#             if is_appending_discontinued_message:
#                 await MessageManager.set_message_history_safely(
#                     user_chat_context=user_chat_context,
#                     new_content=content_buffer,
#                     role=ChatRoles.AI,
#                     index=-1,
#                 )
#             else:
#                 await MessageManager.add_message_history_safely(
#                     user_chat_context=user_chat_context,
#                     content=content_buffer,
#                     role=ChatRoles.AI,
#                 )
#                 is_appending_discontinued_message = True
#             user_chat_context.optional_info["is_discontinued"] = True
#             continue
#         except ChatException as chat_exception:
#             api_logger.error(f"ai exception: {chat_exception.msg}")
#             await MessageManager.rpop_message_history_safely(user_chat_context=user_chat_context, role=ChatRoles.USER)
#             yield chat_exception.msg
#             break
#         except Exception as exception:
#             api_logger.error(f"unexpected ai exception: {exception}", exc_info=True)
#             await MessageManager.rpop_message_history_safely(user_chat_context=user_chat_context, role=ChatRoles.USER)
#             yield "Internal Server Error"
#             break
#         else:
#             await MessageManager.add_message_history_safely(
#                 user_chat_context=user_chat_context, content=content_buffer, role=ChatRoles.AI
#             )
#             user_chat_context.optional_info["is_discontinued"] = False
#             break
