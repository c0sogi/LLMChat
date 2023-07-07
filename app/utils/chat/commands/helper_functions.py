from uuid import uuid4

from app.models.chat_models import UserChatContext
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.cache import CacheManager

from inspect import Parameter, iscoroutinefunction, signature
from types import NoneType
from typing import Any, Callable, Optional, Tuple, Union, get_args, get_origin
from uuid import uuid4

from fastapi import WebSocket
from fastapi.concurrency import run_in_threadpool

from app.errors.api_exceptions import InternalServerError
from app.models.chat_models import (
    ResponseType,
    UserChatContext,
)
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.cache import CacheManager
from app.utils.chat.messages.handler import MessageHandler
from app.utils.chat.managers.websocket import SendToWebsocket


async def create_new_chat_room(
    user_id: str,
    new_chat_room_id: str | None = None,
    buffer: BufferedUserContext | None = None,
) -> UserChatContext:
    if buffer is not None:
        default: UserChatContext = UserChatContext.construct_default(
            user_id=user_id,
            chat_room_id=new_chat_room_id if new_chat_room_id else uuid4().hex,
            llm_model=buffer.current_llm_model,
        )
    else:
        default: UserChatContext = UserChatContext.construct_default(
            user_id=user_id,
            chat_room_id=new_chat_room_id if new_chat_room_id else uuid4().hex,
        )
    await CacheManager.create_context(user_chat_context=default)
    if buffer is not None:
        buffer.insert_context(user_chat_context=default)
        await buffer.change_context_to(index=0)
    return default


async def delete_chat_room(
    chat_room_id_to_delete: str,
    buffer: BufferedUserContext,
) -> bool:
    await CacheManager.delete_chat_room(
        user_id=buffer.user_id, chat_room_id=chat_room_id_to_delete
    )
    index: int | None = buffer.find_index_of_chatroom(
        chat_room_id=chat_room_id_to_delete
    )
    if index is None:
        return False
    buffer.delete_context(index=index)
    if not buffer:
        await create_new_chat_room(
            user_id=buffer.user_id,
            buffer=buffer,
        )
    if buffer.current_chat_room_id == chat_room_id_to_delete:
        await buffer.change_context_to(index=0)
    return True


async def command_handler(
    callback_name: str,
    callback_args: list[str],
    translate: Optional[str],
    buffer: BufferedUserContext,
    callback_finder: Callable[[str], Callable],
):
    callback_response, response_type = await _get_command_response(
        callback_name=callback_name,
        callback_args=callback_args,
        buffer=buffer,
        translate=translate,
        callback_finder=callback_finder,
    )
    if response_type is ResponseType.DO_NOTHING:
        return
    elif response_type is ResponseType.HANDLE_AI:
        await MessageHandler.ai(
            translate=translate,
            buffer=buffer,
        )
        return
    elif response_type is ResponseType.HANDLE_USER:
        await MessageHandler.user(
            msg=callback_response,
            translate=translate,
            buffer=buffer,
        )
        return
    elif response_type is ResponseType.HANDLE_BOTH:
        await MessageHandler.user(
            msg=callback_response,
            translate=translate,
            buffer=buffer,
        )
        await MessageHandler.ai(
            translate=translate,
            buffer=buffer,
        )
        return
    elif response_type is ResponseType.REPEAT_COMMAND:
        splitted: list[str] = callback_response.split(" ")
        await command_handler(
            callback_name=splitted[0][1:]
            if splitted[0].startswith("/")
            else splitted[0],
            callback_args=splitted[1:],
            translate=translate,
            buffer=buffer,
            callback_finder=callback_finder,
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
                kwargs_to_pass[param.name] = available_annotated[
                    param.annotation
                ]
            else:
                raise InternalServerError()
        elif param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            if param.annotation in available_annotated:
                kwargs_to_pass[param.name] = available_annotated[
                    param.annotation
                ]
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
                        args_to_pass.append(
                            param.annotation(available_args.pop(0))
                        )
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


async def _get_command_response(
    callback_name: str,
    callback_args: list[str],
    buffer: BufferedUserContext,
    callback_finder: Callable[[str], Callable],
    **kwargs: Any,
) -> Tuple[Any, ResponseType]:
    if callback_name.startswith("_"):
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg="Command name cannot start with '_'",
            chat_room_id=buffer.current_chat_room_id,
        )
        return None, ResponseType.DO_NOTHING
    else:
        if callback_finder is None:
            return None, ResponseType.DO_NOTHING
        callback: Callable = callback_finder(callback_name)
    try:
        args_to_pass, kwargs_to_pass = _arguments_provider(
            func=callback,
            available_args=callback_args,
            available_annotated={
                UserChatContext: buffer.current_user_chat_context,
                WebSocket: buffer.websocket,
                BufferedUserContext: buffer,
            },
            available_kwargs=buffer.current_user_chat_context.optional_info
            | kwargs,
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
                    if response_type
                    == ResponseType.SEND_MESSAGE_AND_KEEP_GOING
                    else ResponseType.DO_NOTHING
                )
            return callback_response, response_type
        else:
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=callback_response,
                chat_room_id=buffer.current_chat_room_id,
            )
            return None, ResponseType.DO_NOTHING
