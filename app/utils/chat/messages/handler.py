"""A module for handling messages from Frontend"""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from inspect import Parameter, iscoroutinefunction, signature
from types import NoneType
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from uuid import uuid4

import aiohttp
from fastapi import WebSocket
from fastapi.concurrency import run_in_threadpool
from openai import OpenAIError

from app.common.config import ChatConfig, config
from app.errors.api_exceptions import InternalServerError
from app.errors.chat_exceptions import (
    ChatException,
    ChatFunctionCallException,
    ChatInterruptedException,
    ChatModelNotImplementedException,
    ChatTextGenerationException,
    ChatTooMuchTokenException,
)
from app.models.base_models import StreamProgress, SummarizedResult
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

T = TypeVar("T")


@asynccontextmanager
async def _ai_context_manager(
    buffer: BufferedUserContext,
    stream_progress: StreamProgress,
):
    backup_context: UserChatContext = deepcopy(buffer.current_user_chat_context)
    chat_text_generator_error: Optional[ChatTextGenerationException] = None
    try:
        yield

    except ChatInterruptedException:
        buffer.done.clear()
        raise ChatInterruptedException(
            msg=stream_progress.response + stream_progress.buffer
        )
    except ChatFunctionCallException as function_call_exception:
        await MessageHandler.function_call(
            callback_name=function_call_exception.func_name,
            callback_kwargs=function_call_exception.func_kwargs,
            buffer=buffer,
        )
    except ChatException as chat_exception:
        chat_text_generator_error = ChatTextGenerationException(msg=chat_exception.msg)
    except OpenAIError as e:
        raise ChatTextGenerationException(msg=str(e))
    except (
        aiohttp.ServerTimeoutError,
        aiohttp.ClientPayloadError,
        asyncio.TimeoutError,
    ):
        pass
    except Exception as exception:
        ApiLogger.cerror(f"unexpected chat exception: {exception}", exc_info=True)
        chat_text_generator_error = ChatTextGenerationException(msg="Unknown error")
    finally:
        if chat_text_generator_error is not None:
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
        model_name=model.name,
        wait_next_query=True if buffer.optional_info.get("translate") else None,
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
                    to_summarize=buffer.current_ai_message_histories[-1].content,
                    message_history_uuid=buffer.current_ai_message_histories[-1].uuid,
                )
            )
        )


def _arguments_provider(
    func: Callable,
    available_args: list[Any],
    available_annotated: dict[Any, Any],
    available_kwargs: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    args_to_pass: list[Any] = []
    kwargs_to_pass: dict[str, Any] = {}

    for param in signature(func).parameters.values():
        if param.kind == Parameter.VAR_POSITIONAL:
            args_to_pass.extend(available_args)
        elif param.kind == Parameter.VAR_KEYWORD:
            kwargs_to_pass.update(available_kwargs)
        elif param.kind == Parameter.KEYWORD_ONLY:
            if param.annotation in available_annotated:
                kwargs_to_pass[param.name] = available_annotated[param.annotation]
            else:
                raise InternalServerError()
        elif param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            if param.annotation in available_annotated:
                kwargs_to_pass[param.name] = available_annotated[param.annotation]
            elif param.name in available_kwargs:
                kwargs_to_pass[param.name] = available_kwargs[param.name]
            elif param.default is not Parameter.empty:
                kwargs_to_pass[param.name] = param.default
            else:
                try:
                    if param.annotation is Parameter.empty:
                        kwargs_to_pass[param.name] = available_args.pop(0)
                    else:
                        kwargs_to_pass[param.name] = param.annotation(
                            available_args.pop(0)
                        )
                except IndexError:
                    raise IndexError(
                        f"Required argument {param.name} is missing in available_args"
                    )
                except Exception:
                    raise TypeError(
                        f"Required argument {param.name} is missing in available_annotated"
                    )
        elif param.kind == Parameter.POSITIONAL_ONLY:
            if available_args:
                if param.annotation is str:
                    args_to_pass.append(" ".join(available_args))
                    available_args.clear()
                elif param.annotation is Parameter.empty:
                    args_to_pass.append(available_args.pop(0))
                elif get_origin(param.annotation) is Union:
                    union_args = get_args(param.annotation)
                    if str in union_args:
                        args_to_pass.append(" ".join(available_args))
                        available_args.clear()
                    else:
                        for annotation in union_args:
                            try:
                                args_to_pass.append(
                                    annotation.__init__(available_args.pop(0))
                                )
                                break
                            except Exception:
                                raise TypeError(
                                    f"Required argument {param.name} is missing in available_annotated"
                                )
                else:
                    try:
                        args_to_pass.append(param.annotation(available_args.pop(0)))
                    except Exception:
                        raise TypeError(
                            f"Required argument {param.name} is missing in available_annotated"
                        )
            elif param.default is not Parameter.empty:
                args_to_pass.append(param.default)
            elif get_origin(param.annotation) is Union:
                if NoneType in get_args(param.annotation):
                    args_to_pass.append(None)
                else:
                    raise TypeError(
                        f"Required argument {param.name} is missing in available_args"
                    )
            else:
                raise IndexError(
                    f"Required argument {param.name} is missing in available_args"
                )

    return args_to_pass, kwargs_to_pass


async def _command_interpreter(
    callback_name: str,
    callback_args: list[str],
    callback_kwargs: dict[str, Any],
    callback_finder: Callable[[str], Callable],
    buffer: BufferedUserContext,
) -> Optional[Any]:
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
        print("- DEBUG:>>", callback_response, response_type)
    if response_type is ResponseType.DO_NOTHING:
        return callback_response
    elif response_type is ResponseType.HANDLE_AI:
        await MessageHandler.ai(
            buffer=buffer,
        )
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
        await MessageHandler.ai(
            buffer=buffer,
        )
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


async def _get_command_response(
    callback: Callable,
    callback_args: list[str],
    callback_kwargs: dict[str, Any],
    buffer: BufferedUserContext,
) -> tuple[Any, ResponseType]:
    try:
        args_to_pass, kwargs_to_pass = _arguments_provider(
            func=callback,
            available_args=callback_args,
            available_annotated={
                UserChatContext: buffer.current_user_chat_context,
                WebSocket: buffer.websocket,
                BufferedUserContext: buffer,
            },
            available_kwargs=callback_kwargs,
        )
    except TypeError:
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg="Wrong argument type",
            chat_room_id=buffer.current_chat_room_id,
        )
        return None, ResponseType.DO_NOTHING
    except IndexError:
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg="Not enough arguments",
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
        if isinstance(callback_response, tuple):
            callback_response, response_type = callback_response
            if response_type in (
                ResponseType.SEND_MESSAGE_AND_STOP,
                ResponseType.SEND_MESSAGE_AND_KEEP_GOING,
            ):
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=callback_response,
                    chat_room_id=buffer.current_chat_room_id,
                )
                return callback_response, (
                    ResponseType.HANDLE_BOTH
                    if response_type == ResponseType.SEND_MESSAGE_AND_KEEP_GOING
                    else ResponseType.DO_NOTHING
                )
            return callback_response, response_type
        else:
            return callback_response, ResponseType.DO_NOTHING


async def _interruption_event_watcher(
    future: asyncio.Future[T],
    event: asyncio.Event,
    hold_interruption_event: Optional[asyncio.Event] = None,
) -> T:
    async def monitoring_events():
        while True:
            await event.wait()

            if hold_interruption_event is None or not hold_interruption_event.is_set():
                return
            await asyncio.sleep(0.1)

    done, pending = await asyncio.wait(
        [
            future,
            asyncio.ensure_future(monitoring_events()),
        ],
        return_when=asyncio.FIRST_COMPLETED,  # Return after the first one completes
    )
    for t in pending:
        t.cancel()
    for t in done:
        if t is future:
            return t.result()  # type: ignore
    raise ChatInterruptedException()


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
                    to_summarize=buffer.current_user_message_histories[-1].content,
                    message_history_uuid=buffer.current_user_message_histories[-1].uuid,
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
        content=await run_in_threadpool(get_summarization, to_summarize=to_summarize),
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
        if not buffer.current_user_message_histories and UTC.check_string_valid(
            buffer.current_chat_room_name
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
            else current_llm_model.max_total_tokens - ChatConfig.extra_token_margin
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
            calculated_tokens_to_use=user_token + current_llm_model.token_margin,
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
        async with _ai_context_manager(buffer=buffer, stream_progress=stream_progress):
            await _interruption_event_watcher(
                future=asyncio.ensure_future(
                    _ai_stream(
                        model=model,
                        buffer=buffer,
                        stream_progress=stream_progress,
                    )
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
        await _interruption_event_watcher(
            future=asyncio.ensure_future(
                _command_interpreter(
                    callback_name=callback_name,
                    callback_args=callback_args,
                    callback_kwargs=buffer.optional_info,
                    callback_finder=callback_finder,
                    buffer=buffer,
                )
            ),
            event=buffer.done,
        )

    @classmethod
    async def function_call(
        cls,
        callback_name: str,
        callback_kwargs: dict[str, Any],
        buffer: BufferedUserContext,
    ) -> None:
        result = await _interruption_event_watcher(
            future=asyncio.ensure_future(
                _command_interpreter(
                    callback_name=callback_name,
                    callback_args=[],
                    callback_kwargs=callback_kwargs | buffer.optional_info,
                    callback_finder=FunctionCalls._get_function,
                    buffer=buffer,
                )
            ),
            event=buffer.done,
        )
        if result:
            await MessageManager.add_message_history_safely(
                user_chat_context=buffer.current_user_chat_context,
                content=str(result),
                role=ChatRoles.SYSTEM,
            )
