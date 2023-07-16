"""A module for handling messages from Frontend"""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from inspect import iscoroutinefunction
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    TypeVar,
)
from uuid import uuid4

import aiohttp
from fastapi import WebSocket
from fastapi.concurrency import run_in_threadpool
from openai import OpenAIError

from app.common.config import ChatConfig, config
from app.errors.chat_exceptions import (
    ChatException,
    ChatFunctionCallException,
    ChatInterruptedException,
    ChatModelNotImplementedException,
    ChatTextGenerationException,
    ChatTooMuchTokenException,
)
from app.models.base_models import (
    MessageHistory,
    StreamProgress,
    SummarizedResult,
)
from app.models.chat_models import ChatRoles, ResponseType, UserChatContext
from app.models.function_calling.functions import FunctionCalls
from app.models.llms import ExllamaModel, LlamaCppModel, LLMModel, OpenAIModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.cache import CacheManager
from app.utils.chat.managers.message import MessageManager
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.chat.text_generations.completion_api import (
    agenerate_from_chat_completion_api,
    agenerate_from_text_completion_api,
)
from app.utils.chat.text_generations.summarization import get_summarization
from app.utils.date_utils import UTC
from app.utils.function_calling.callbacks.translate import translate_callback
from app.utils.logger import ApiLogger
from app.utils.types import arguments_provider

T = TypeVar("T")


async def _save_function_call_result(
    buffer: BufferedUserContext,
    function_call_name: str,
    function_call_result: str,
    update_cache: bool = True,
) -> None:
    """Save function call result to both buffer and database"""
    if update_cache:
        await MessageManager.add_message_history_safely(
            user_chat_context=buffer.current_user_chat_context,
            content=function_call_result,
            role=ChatRoles.SYSTEM,
            custom_role=f"function: {function_call_name}",
        )
    else:
        buffer.current_system_message_histories.append(
            MessageHistory(
                role=f"function: {function_call_name}",
                content=function_call_result,
                tokens=buffer.current_user_chat_context.get_tokens_of(
                    function_call_result
                ),
                actual_role=ChatRoles.SYSTEM.value,
            )
        )


@asynccontextmanager
async def _ai_context_manager(
    buffer: BufferedUserContext,
    model: LLMModel,
    stream_progress: StreamProgress,
):
    backup_context: UserChatContext = deepcopy(
        buffer.current_user_chat_context
    )
    chat_text_generator_error: Optional[ChatTextGenerationException] = None
    try:
        # Hand shake with frontend
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=None,
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
            model_name=model.name,
            uuid=stream_progress.uuid,
        )
        yield

    except ChatInterruptedException:
        # Raise ChatInterruptedException to stop the stream,
        # with generated text so far.
        # The EOS will be sent to frontend in `finally` block
        buffer.done.clear()
        raise ChatInterruptedException(
            msg=stream_progress.response + stream_progress.buffer
        )
    except ChatFunctionCallException as function_call_exception:
        # Handle function call exception
        await MessageHandler.function_call(
            callback_name=function_call_exception.func_name,
            callback_kwargs=function_call_exception.func_kwargs,
            buffer=buffer,
        )
    except ChatException as chat_exception:
        chat_text_generator_error = ChatTextGenerationException(
            msg=chat_exception.msg
        )
    except OpenAIError as e:
        raise ChatTextGenerationException(msg=str(e))
    except (
        aiohttp.ServerTimeoutError,
        aiohttp.ClientPayloadError,
        asyncio.TimeoutError,
    ):
        pass
    except Exception as exception:
        ApiLogger.cerror(
            f"unexpected chat exception: {exception}", exc_info=True
        )
        chat_text_generator_error = ChatTextGenerationException(
            msg="Unknown error"
        )
    finally:
        if chat_text_generator_error is not None:
            # Restore context, and send error message to frontend

            buffer.current_user_chat_context.copy_from(backup_context)
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=(
                    "\n\nAn error occurred while generating text: "
                    f"**{chat_text_generator_error.msg}**"
                ),
                chat_room_id=buffer.current_chat_room_id,
                finish=True,
                model_name=buffer.current_user_chat_context.llm_model.value.name,
            )
        else:
            # Send EOS to frontend

            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=stream_progress.buffer,
                chat_room_id=buffer.current_chat_room_id,
                finish=True,
                model_name=model.name,
                wait_next_query=False,
            )


async def _ai_stream(
    model: LLMModel,
    buffer: BufferedUserContext,
    stream_progress: StreamProgress,
) -> None:
    if isinstance(model, OpenAIModel):
        stream_func = agenerate_from_chat_completion_api

    elif isinstance(model, (LlamaCppModel, ExllamaModel)):
        if config.is_llama_available and config.llama_completion_url:
            # Use llama_cpp API
            if "/v1/chat/completions" in config.llama_completion_url:
                stream_func = agenerate_from_chat_completion_api
            elif "/v1/completions" in config.llama_completion_url:
                stream_func = agenerate_from_text_completion_api
            else:
                raise ChatModelNotImplementedException(
                    msg=f"Model {model.name} not implemented."
                )
        else:
            if config.is_llama_booting:
                raise ChatModelNotImplementedException(
                    msg=f"Model {model.name} is currently booting."
                )
            raise ChatModelNotImplementedException(
                msg=f"Model {model.name} not available."
            )
    else:
        raise ChatModelNotImplementedException(
            msg=f"Model {buffer.current_user_chat_context.llm_model.value.name} not implemented."
        )
    await SendToWebsocket.stream(
        buffer=buffer,
        stream_func=stream_func,
        stream_progress=stream_progress,
    )


async def _ai_summarization(buffer: BufferedUserContext) -> None:
    if (
        ChatConfig.summarize_for_chat
        and buffer.current_ai_message_histories[-1].tokens
        > ChatConfig.summarization_threshold
    ):
        buffer.task_list.append(
            asyncio.create_task(
                summarization_task(
                    user_id=buffer.user_id,
                    chat_room_id=buffer.current_chat_room_id,
                    role="ai",
                    to_summarize=buffer.current_ai_message_histories[
                        -1
                    ].content,
                    message_history_uuid=buffer.current_ai_message_histories[
                        -1
                    ].uuid,
                )
            )
        )


async def _command_interpreter(
    callback_name: str,
    callback_args: list[str],
    callback_kwargs: dict[str, Any],
    callback_finder: Callable[[str], Callable],
    buffer: BufferedUserContext,
) -> Optional[Any]:
    """Interpret command, which is found by callback_finder,
    and call corresponding callback function with arguments provided by
    callback_args and callback_kwargs, and handle the callback response.
    Finally, return the callback response."""

    if callback_name.startswith("_"):
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg="Command name cannot start with '_'",
            chat_room_id=buffer.current_chat_room_id,
        )
        callback_response, response_type = None, ResponseType.DO_NOTHING
    else:
        callback_response, response_type = await _get_command_response(
            callback=callback_finder(callback_name),
            callback_args=callback_args,
            callback_kwargs=callback_kwargs,
            buffer=buffer,
        )

    if response_type is ResponseType.DO_NOTHING:
        return callback_response

    elif response_type is ResponseType.HANDLE_AI:
        await MessageHandler.ai(buffer=buffer)
        return callback_response

    elif response_type is ResponseType.HANDLE_USER:
        await MessageHandler.user(
            msg=callback_response or "",
            buffer=buffer,
        )
        return callback_response

    elif response_type is ResponseType.HANDLE_BOTH:
        await MessageHandler.user(
            msg=callback_response or "",
            buffer=buffer,
        )
        await MessageHandler.ai(buffer=buffer)
        return callback_response

    elif response_type is ResponseType.REPEAT_COMMAND and callback_response:
        splitted: list[str] = callback_response.split(" ")
        await _command_interpreter(
            callback_name=splitted[0][1:]
            if splitted[0].startswith("/")
            else splitted[0],
            callback_args=splitted[1:],
            callback_kwargs=callback_kwargs,
            callback_finder=callback_finder,
            buffer=buffer,
        )

    return None


@asynccontextmanager
async def _function_call_context_manager(buffer: BufferedUserContext):
    # Do not invoke function call again
    buffer.optional_info["function_call"] = "none"
    yield
    # Remove function call messages when all function calls are done
    buffer.current_system_message_histories[:] = [
        system_message_history
        for system_message_history in buffer.current_system_message_histories
        if not system_message_history.role.startswith("function:")
    ]
    # Restore original function call
    buffer.optional_info["functions"] = None
    buffer.optional_info["function_call"] = None


async def _get_command_response(
    callback: Callable,
    callback_args: list[str],
    callback_kwargs: dict[str, Any],
    buffer: BufferedUserContext,
) -> tuple[Any, ResponseType]:
    try:
        args_to_pass, kwargs_to_pass = arguments_provider(
            func=callback,
            available_args=callback_args,
            available_annotated={
                UserChatContext: buffer.current_user_chat_context,
                WebSocket: buffer.websocket,
                BufferedUserContext: buffer,
            },
            available_kwargs=callback_kwargs,
        )
    except Exception as e:
        ApiLogger.cerror(f"Error in arguments_provider: {e}", exc_info=True)
        if isinstance(e, (ValueError, TypeError, IndexError)):
            error_msg = str(e)
        else:
            error_msg = "Unexpected error while parsing arguments!"
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=error_msg,
            chat_room_id=buffer.current_chat_room_id,
        )
        return None, ResponseType.DO_NOTHING
    else:
        if iscoroutinefunction(callback):  # if callback is coroutine function
            callback_response = await callback(*args_to_pass, **kwargs_to_pass)
        else:
            callback_response = await run_in_threadpool(
                callback, *args_to_pass, **kwargs_to_pass
            )
        if (
            isinstance(callback_response, tuple)
            and len(callback_response) == 2
        ):
            callback_response, response_type = callback_response
            if response_type is ResponseType.SEND_MESSAGE_AND_STOP:
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=callback_response,
                    chat_room_id=buffer.current_chat_room_id,
                )
                return callback_response, ResponseType.DO_NOTHING
            return callback_response, response_type
        else:
            return callback_response, ResponseType.DO_NOTHING


async def _interruption_event_watcher(
    coro_or_future: Awaitable[T],
    event: asyncio.Event,
    hold_interruption_event: Optional[asyncio.Event] = None,
) -> T:
    """Watch interruption event, and raise ChatInterruptedException if it's set
    while waiting for coro_or_future to complete. If hold_interruption_event is set,
    ChatInterruptedException will not be raised until hold_interruption_event is unset.
    """

    async def monitoring_events():  # Monitor interruption event
        while True:
            await event.wait()  # Wait for interruption event

            if (
                hold_interruption_event is None
                or not hold_interruption_event.is_set()
            ):
                # If hold_interruption_event is not given, or it's not set,
                # raise ChatInterruptedException immediately
                return
            # If hold_interruption_event is set, loop until it's unset
            await asyncio.sleep(0.1)

    future = asyncio.ensure_future(coro_or_future)
    done, pending = await asyncio.wait(
        [
            future,
            asyncio.ensure_future(monitoring_events()),
        ],
        return_when=asyncio.FIRST_COMPLETED,  # Return after the first one completes
    )
    for pending_task in pending:
        pending_task.cancel()  # Cancel the other one
    for done_task in done:
        if (
            done_task is future
        ):  # If coro_or_future completes first, return its result
            return done_task.result()  # type: ignore
    raise ChatInterruptedException()  # If interruption event completes first, raise ChatInterruptedException


async def _user_summarization(buffer: BufferedUserContext) -> None:
    if (
        ChatConfig.summarize_for_chat
        and buffer.current_user_message_histories[-1].tokens
        > ChatConfig.summarization_threshold
    ):
        buffer.task_list.append(
            asyncio.create_task(
                summarization_task(
                    user_id=buffer.user_id,
                    chat_room_id=buffer.current_chat_room_id,
                    role="user",
                    to_summarize=buffer.current_user_message_histories[
                        -1
                    ].content,
                    message_history_uuid=buffer.current_user_message_histories[
                        -1
                    ].uuid,
                )
            )
        )


async def summarization_task(
    user_id: str,
    chat_room_id: str,
    role: str,
    to_summarize: str,
    message_history_uuid: str,
) -> SummarizedResult:  # =
    return SummarizedResult(
        user_id=user_id,
        chat_room_id=chat_room_id,
        role=role,
        content=await run_in_threadpool(
            get_summarization, to_summarize=to_summarize
        ),
        uuid=message_history_uuid,
    )


class MessageHandler:
    @staticmethod
    async def user(
        msg: str,
        buffer: BufferedUserContext,
        use_tight_token_limit: bool = True,
    ) -> None:
        """Handle user message, including translation"""
        if (
            not buffer.current_user_message_histories
            and UTC.check_string_valid(buffer.current_chat_room_name)
        ):
            buffer.current_chat_room_name = msg[:20]
            await CacheManager.update_profile(
                user_chat_profile=buffer.current_user_chat_profile,
            )
            await SendToWebsocket.init(
                buffer=buffer, send_chat_rooms=True, wait_next_query=True
            )
        translate: Optional[str] = buffer.optional_info.get("translate")
        if translate:  # if user message is translated
            translate_chain_result: Optional[str] = await translate_callback(
                buffer=buffer,
                query=msg,
                finish=False,
                wait_next_query=False,
                show_result=True,
                src_lang=translate,
                trg_lang="en",
            )
            if translate_chain_result is not None:
                msg = translate_chain_result

        user_token: int = buffer.current_user_chat_context.get_tokens_of(msg)
        current_llm_model: LLMModel = buffer.current_llm_model.value
        token_limit: int = (
            current_llm_model.max_tokens_per_request
            if use_tight_token_limit
            else current_llm_model.max_total_tokens
            - ChatConfig.extra_token_margin
        )
        if user_token > token_limit:  # if user message is too long
            raise ChatTooMuchTokenException(
                msg=f"Message too long. Now {user_token} tokens, "
                f"but {token_limit} tokens allowed."
            )
        await MessageManager.add_message_history_safely(
            user_chat_context=buffer.current_user_chat_context,
            content=msg,
            role=ChatRoles.USER,
            uuid=buffer.optional_info.get("uuid"),
            calculated_tokens_to_use=user_token
            + current_llm_model.token_margin,
        )
        await _user_summarization(buffer=buffer)

    @classmethod
    async def ai(
        cls,
        buffer: BufferedUserContext,
        model: Optional[LLMModel] = None,
    ) -> None:
        """Handle ai message, including text generation and translation"""
        stream_progress = StreamProgress(uuid=uuid4().hex)
        if model is None:
            model = buffer.current_llm_model.value
        async with _ai_context_manager(
            buffer=buffer, model=model, stream_progress=stream_progress
        ):
            await _interruption_event_watcher(
                _ai_stream(
                    model=model, buffer=buffer, stream_progress=stream_progress
                ),
                event=buffer.done,
            )
            await MessageManager.add_message_history_safely(
                user_chat_context=buffer.current_user_chat_context,
                content=stream_progress.response,
                role=ChatRoles.AI,
                uuid=stream_progress.uuid,
            )
            await _ai_summarization(buffer=buffer)
            translate: Optional[str] = buffer.optional_info.get("translate")
            if translate:
                await translate_callback(
                    buffer=buffer,
                    query=stream_progress.response,
                    finish=True,
                    wait_next_query=False,
                    show_result=True,
                    src_lang="en",
                    trg_lang=translate,
                )

    @classmethod
    async def command(
        cls,
        callback_name: str,
        callback_args: list[str],
        callback_finder: Callable[[str], Callable],
        buffer: BufferedUserContext,
    ) -> None:
        await _command_interpreter(
            callback_name=callback_name,
            callback_args=callback_args,
            callback_kwargs=dict(buffer.optional_info),
            callback_finder=callback_finder,
            buffer=buffer,
        )

    @classmethod
    async def function_call(
        cls,
        callback_name: str,
        callback_kwargs: dict[str, Any],
        buffer: BufferedUserContext,
    ) -> None:
        print(
            f"- DEBUG: Handling function call {callback_name} with {callback_kwargs}",
            flush=True,
        )
        async with _function_call_context_manager(buffer=buffer):
            function_call_callback_result: Optional[
                Any
            ] = await _command_interpreter(
                callback_name=callback_name,
                callback_args=[],
                callback_kwargs=callback_kwargs,
                callback_finder=FunctionCalls.get_function,
                buffer=buffer,
            )
            if function_call_callback_result:
                await _save_function_call_result(
                    buffer=buffer,
                    function_call_name=callback_name,
                    function_call_result=str(function_call_callback_result),
                    update_cache=False,
                )
            await MessageHandler.ai(buffer=buffer)
