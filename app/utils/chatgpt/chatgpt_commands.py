from asyncio import gather
from enum import Enum
from inspect import Parameter, iscoroutinefunction, signature
from typing import Any, Callable, Tuple
from uuid import uuid4
from functools import wraps
from fastapi import WebSocket

from app.errors.api_exceptions import InternalServerError
from app.utils.chatgpt.chatgpt_cache_manager import chatgpt_cache_manager
from app.utils.chatgpt.chatgpt_message_manager import MessageManager
from app.utils.chatgpt.chatgpt_websocket_manager import SendToWebsocket
from app.utils.chatgpt.chatgpt_buffer import BufferedUserContext
from app.viewmodels.gpt_models import GptRoles, MessageHistory, OpenAIModels, UserGptContext


class ResponseType(str, Enum):
    SEND_MESSAGE_AND_STOP = "send_message_and_stop"
    SEND_MESSAGE_AND_KEEP_GOING = "send_message_and_keep_going"
    HANDLE_USER = "handle_user"
    HANDLE_GPT = "handle_gpt"
    HANDLE_BOTH = "handle_both"
    DO_NOTHING = "do_nothing"
    REPEAT_COMMAND = "repeat_command"


class CommandResponse:
    @staticmethod
    def _wrapper(enum_type: ResponseType) -> Callable:
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, ResponseType]:
                result = func(*args, **kwargs)
                return (result, enum_type)

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, ResponseType]:
                result = await func(*args, **kwargs)
                return (result, enum_type)

            return async_wrapper if iscoroutinefunction(func) else sync_wrapper

        return decorator

    send_message_and_stop = _wrapper(ResponseType.SEND_MESSAGE_AND_STOP)
    send_message_and_keep_going = _wrapper(ResponseType.SEND_MESSAGE_AND_KEEP_GOING)
    handle_user = _wrapper(ResponseType.HANDLE_USER)
    handle_gpt = _wrapper(ResponseType.HANDLE_GPT)
    handle_both = _wrapper(ResponseType.HANDLE_BOTH)
    do_nothing = _wrapper(ResponseType.DO_NOTHING)
    repeat_command = _wrapper(ResponseType.REPEAT_COMMAND)


async def create_new_chat_room(
    user_id: str,
    new_chat_room_id: str | None = None,
    buffer: BufferedUserContext | None = None,
) -> UserGptContext:
    default: UserGptContext = UserGptContext.construct_default(
        user_id=user_id,
        chat_room_id=new_chat_room_id if new_chat_room_id else uuid4().hex,
    )
    await chatgpt_cache_manager.create_context(user_gpt_context=default)
    if buffer is not None:
        buffer.insert_context(user_gpt_context=default)
        buffer.change_context_to(index=0)
    return default


async def delete_chat_room(
    user_id: str,
    chat_room_id: str,
    buffer: BufferedUserContext | None = None,
) -> None:
    await chatgpt_cache_manager.delete_chat_room(user_id=user_id, chat_room_id=chat_room_id)
    if buffer is not None:
        index: int | None = buffer.find_index_of_chatroom(chat_room_id=chat_room_id)
        if index is not None:
            buffer.delete_context(index=index)
            if buffer.buffer_size == 0:
                await create_new_chat_room(
                    user_id=user_id,
                    buffer=buffer,
                )
            buffer.change_context_to(index=0)


async def get_contexts_sorted_from_recent_to_past(user_id: str, chat_room_ids: list[str]) -> list[UserGptContext]:
    if len(chat_room_ids) == 0:
        # create new chatroom
        return [await create_new_chat_room(user_id=user_id)]
    else:
        # get latest chatroom
        contexts: list[UserGptContext] = await gather(
            *[
                chatgpt_cache_manager.read_context(user_id=user_id, chat_room_id=chat_room_id)
                for chat_room_id in chat_room_ids
            ]
        )
        contexts.sort(key=lambda x: x.user_gpt_profile.created_at, reverse=True)
        return contexts


def arguments_provider(
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
            elif param.default is not Parameter.empty:
                kwargs_to_pass[param.name] = param.default
            else:
                try:
                    if param.annotation is Parameter.empty:
                        kwargs_to_pass[param.name] = available_args.pop(0)
                    else:
                        kwargs_to_pass[param.name] = param.annotation(available_args.pop(0))
                except IndexError:
                    raise IndexError(f"Required argument {param.name} is missing in available_args")
                except Exception:
                    raise TypeError(f"Required argument {param.name} is missing in available_annotated")
        elif param.kind == Parameter.POSITIONAL_ONLY:
            if len(available_args) > 0:
                if param.annotation is str:
                    args_to_pass.append(" ".join(available_args))
                    available_args.clear()
                elif param.annotation is Parameter.empty:
                    args_to_pass.append(available_args.pop(0))
                else:
                    try:
                        args_to_pass.append(param.annotation(available_args.pop(0)))
                    except Exception:
                        raise TypeError(f"Required argument {param.name} is missing in available_annotated")
            elif param.default is not Parameter.empty:
                args_to_pass.append(param.default)
            else:
                raise IndexError(f"Required argument {param.name} is missing in available_args")
    return args_to_pass, kwargs_to_pass


class ChatGptCommands:  # commands for chat gpt
    @classmethod
    async def _get_command_response(
        cls,
        callback_name: str,
        callback_args: list[str],
        buffer: BufferedUserContext,
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
            callback: Callable = getattr(cls, callback_name, cls.not_existing_callback)  # get callback function
        try:
            args_to_pass, kwargs_to_pass = arguments_provider(
                func=callback,
                available_args=callback_args,
                available_annotated={
                    UserGptContext: buffer.current_user_gpt_context,
                    WebSocket: buffer.websocket,
                    BufferedUserContext: buffer,
                },
                available_kwargs=buffer.current_user_gpt_context.optional_info | kwargs,
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
                callback_response = callback(*args_to_pass, **kwargs_to_pass)
            if isinstance(callback_response, tuple):
                callback_response, response_type = callback_response
                if response_type in (ResponseType.SEND_MESSAGE_AND_STOP, ResponseType.SEND_MESSAGE_AND_KEEP_GOING):
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
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=callback_response,
                    chat_room_id=buffer.current_chat_room_id,
                )
                return None, ResponseType.DO_NOTHING

    @staticmethod
    @CommandResponse.send_message_and_stop
    def not_existing_callback() -> str:  # callback for not existing command
        return "Sorry, I don't know what you mean by..."

    @classmethod
    @CommandResponse.send_message_and_stop
    def help(cls) -> str:
        docs: list[str] = [
            getattr(cls, callback_name).__doc__ for callback_name in dir(cls) if not callback_name.startswith("_")
        ]
        return "\n\n".join([doc for doc in docs if doc is not None])

    @staticmethod
    @CommandResponse.do_nothing
    async def deletechatroom(buffer: BufferedUserContext) -> None:
        await delete_chat_room(
            user_id=buffer.user_id,
            chat_room_id=buffer.current_chat_room_id,
            buffer=buffer,
        )
        await SendToWebsocket.initiation_of_chat(websocket=buffer.websocket, buffer=buffer)

    @staticmethod
    @CommandResponse.send_message_and_stop
    async def clear(user_gpt_context: UserGptContext) -> str:  # clear user and gpt message histories
        """Clear user and gpt message histories, and return the number of tokens removed\n
        /clear"""
        n_user_tokens: int = user_gpt_context.user_message_tokens
        n_gpt_tokens: int = user_gpt_context.gpt_message_tokens
        n_system_tokens: int = user_gpt_context.system_message_tokens
        for role in GptRoles:
            getattr(user_gpt_context, f"{role.name.lower()}_message_histories").clear()
            setattr(user_gpt_context, f"{role.name.lower()}_message_tokens", 0)
            await chatgpt_cache_manager.delete_message_history(
                user_id=user_gpt_context.user_id,
                chat_room_id=user_gpt_context.chat_room_id,
                role=role,
            )
        response: str = f"""## Total Token Removed: **{n_user_tokens + n_gpt_tokens + n_system_tokens}**
- User: {n_user_tokens}
- GPT: {n_gpt_tokens}
- System: {n_system_tokens}"""
        return response  # return success message

    @staticmethod
    @CommandResponse.send_message_and_stop
    def test(user_gpt_context: UserGptContext) -> str:  # test command showing user_gpt_context
        """Test command showing user_gpt_context\n
        /test"""
        return str(user_gpt_context)

    @staticmethod
    @CommandResponse.send_message_and_stop
    async def reset(user_gpt_context: UserGptContext) -> str:  # reset user_gpt_context
        """Reset user_gpt_context\n
        /reset"""
        user_gpt_context.reset()
        if await chatgpt_cache_manager.reset_context(
            user_id=user_gpt_context.user_id,
            chat_room_id=user_gpt_context.chat_room_id,
        ):  # if reset success
            return "Context reset success"
        else:
            return "Context reset failed"

    @staticmethod
    @CommandResponse.send_message_and_stop
    async def system(system_message: str, /, user_gpt_context: UserGptContext) -> str:  # add system message
        """Add system message\n
        /system <system_message>"""
        await MessageManager.add_message_history_safely(
            user_gpt_context=user_gpt_context,
            content=system_message,
            role=user_gpt_context.user_gpt_profile.system_role,
        )
        return f"Added system message: {system_message}"  # return success message

    @staticmethod
    @CommandResponse.send_message_and_stop
    async def settemperature(temp_to_change: float, user_gpt_context: UserGptContext) -> str:  # set temperature of gpt
        """Set temperature of gpt\n
        /settemperature <temp_to_change>"""
        try:
            assert 0 <= temp_to_change <= 1  # assert temperature is between 0 and 1
        except AssertionError:  # if temperature is not between 0 and 1
            return "Temperature must be between 0 and 1"
        else:
            previous_temperature: str = str(user_gpt_context.user_gpt_profile.temperature)
            user_gpt_context.user_gpt_profile.temperature = temp_to_change
            await chatgpt_cache_manager.update_profile_and_model(user_gpt_context)  # update user_gpt_context
            return (
                f"I've changed temperature from {previous_temperature} to {temp_to_change}."  # return success message
            )

    @classmethod
    async def temp(cls, temp_to_change: float, user_gpt_context: UserGptContext) -> str:  # alias for settemperature
        """Alias for settemperature\n
        /temp <temp_to_change>"""
        return await cls.settemperature(temp_to_change, user_gpt_context)

    @staticmethod
    @CommandResponse.send_message_and_stop
    async def settopp(top_p_to_change: float, user_gpt_context: UserGptContext) -> str:  # set top_p of gpt
        """Set top_p of gpt\n
        /settopp <top_p_to_change>"""
        try:
            assert 0 <= top_p_to_change <= 1  # assert top_p is between 0 and 1
        except AssertionError:  # if top_p is not between 0 and 1
            return "Top_p must be between 0 and 1."  # return fail message
        else:
            previous_top_p: str = str(user_gpt_context.user_gpt_profile.top_p)
            user_gpt_context.user_gpt_profile.top_p = top_p_to_change  # set top_p
            await chatgpt_cache_manager.update_profile_and_model(user_gpt_context)  # update user_gpt_context
            return f"I've changed top_p from {previous_top_p} to {top_p_to_change}."  # return success message

    @classmethod
    async def topp(cls, top_p_to_change: float, user_gpt_context: UserGptContext) -> str:  # alias for settopp
        """Alias for settopp\n
        /topp <top_p_to_change>"""
        return await cls.settopp(top_p_to_change, user_gpt_context)

    @staticmethod
    @CommandResponse.send_message_and_stop
    async def poplastmessage(role: str, user_gpt_context: UserGptContext) -> str:
        """Pop last message (user or system or gpt)\n
        /poplastmessage <user|system|gpt>"""
        if role.upper() not in GptRoles._member_names_:
            return "Role must be one of user, system, gpt"  # return fail message
        last_message_history: MessageHistory | None = await MessageManager.rpop_message_history_safely(
            user_gpt_context=user_gpt_context, role=role
        )  # pop last message history
        if last_message_history is None:  # if last_message_history is None
            return f"There is no {role} message to pop."  # return fail message
        return f"Pop {role} message: {last_message_history.content}"  # return success message

    @classmethod
    async def pop(cls, role: str, user_gpt_context: UserGptContext) -> str:
        """Alias for poplastmessage\n
        /pop"""
        return await cls.poplastmessage(role, user_gpt_context)

    @staticmethod
    async def retry(user_gpt_context: UserGptContext) -> Tuple[str | None, ResponseType]:
        """Retry last message\n
        /retry"""
        if len(user_gpt_context.user_message_histories) < 1 or len(user_gpt_context.gpt_message_histories) < 1:
            return "There is no message to retry.", ResponseType.SEND_MESSAGE_AND_STOP
        await MessageManager.rpop_message_history_safely(user_gpt_context=user_gpt_context, role=GptRoles.GPT)
        return None, ResponseType.HANDLE_GPT

    @staticmethod
    @CommandResponse.send_message_and_stop
    def ping() -> str:
        """Ping! Pong!\n
        /ping"""
        return "pong"

    @staticmethod
    @CommandResponse.send_message_and_stop
    async def codex(user_gpt_context: UserGptContext) -> str:
        """Let GPT act as CODEX("COding DEsign eXpert")\n
        /codex"""
        system_message = """Act as CODEX ("COding DEsign eXpert"), an expert coder with experience in multiple coding languages.
Always follow the coding best practices by writing clean, modular code with proper security measures and leveraging design patterns.
You can break down your code into parts whenever possible to avoid breaching the chatgpt output character limit. Write code part by part when I send "continue". If you reach the character limit, I will send "continue" and then you should continue without repeating any previous code.
Do not assume anything from your side; please ask me a numbered list of essential questions before starting.
If you have trouble fixing a bug, ask me for the latest code snippets for reference from the official documentation.
I am using [MacOS], [VSCode] and prefer [brew] package manager.
Start a conversation as "CODEX: Hi, what are we coding today?"
        """
        await MessageManager.clear_message_history_safely(user_gpt_context=user_gpt_context, role=GptRoles.SYSTEM)
        await MessageManager.add_message_history_safely(
            user_gpt_context=user_gpt_context, role=GptRoles.SYSTEM, content=system_message
        )
        return "CODEX mode ON"

    @staticmethod
    @CommandResponse.send_message_and_stop
    async def redx(user_gpt_context: UserGptContext) -> str:
        """Let GPT reduce your message as much as possible\n
        /redx"""
        system_message = """compress the following text in a way that fits in a tweet (ideally) and such that you (GPT) can reconstruct the intention of the human who wrote text as close as possible to the original intention. This is for yourself. It does not need to be human readable or understandable. Abuse of language mixing, abbreviations, symbols (unicode and emoji), or any other encodings or internal representations is all permissible, as long as it, if pasted in a new inference cycle, will yield near-identical results as the original text: """
        await MessageManager.clear_message_history_safely(user_gpt_context=user_gpt_context, role=GptRoles.SYSTEM)
        await MessageManager.add_message_history_safely(
            user_gpt_context=user_gpt_context, role=GptRoles.SYSTEM, content=system_message
        )
        return "REDX mode ON"

    @staticmethod
    @CommandResponse.send_message_and_stop
    def echo(msg: str, /) -> str:
        """Echo your message\n
        /echo <msg>"""
        return msg

    @staticmethod
    @CommandResponse.do_nothing
    async def echo2(msg: str, /, websocket: WebSocket, user_gpt_context: UserGptContext) -> None:
        """Send all messages to websocket\n
        /sendtowebsocket"""
        await SendToWebsocket.message(
            websocket=websocket,
            msg=msg,
            chat_room_id=user_gpt_context.chat_room_id,
        )

    @staticmethod
    @CommandResponse.send_message_and_stop
    def codeblock(language, codes: str, /) -> str:
        """Send codeblock\n
        /codeblock <language> <codes>"""
        return f"```{language.lower()}\n" + codes + "\n```"

    @staticmethod
    @CommandResponse.send_message_and_stop
    async def changemodel(model: str, user_gpt_context: UserGptContext) -> str:
        """Change GPT model\n
        /changemodel <model>"""
        if model not in OpenAIModels._member_names_:
            return f"Model must be one of {', '.join(OpenAIModels._member_names_)}"
        llm_model: OpenAIModels = OpenAIModels._member_map_[model]  # type: ignore
        user_gpt_context.gpt_model = llm_model
        await chatgpt_cache_manager.update_profile_and_model(user_gpt_context)
        return f"Model changed to {model}. Actual model: {llm_model.value.name}"

    @staticmethod
    @CommandResponse.send_message_and_stop
    def addoptionalinfo(key: str, value: str, user_gpt_context: UserGptContext) -> str:
        """Add optional info to buffer\n
        /addoptionalinfo <key> <value>"""
        user_gpt_context.optional_info[key] = " ".join(value)
        return f"I've added {key}={value} to your optional info."

    @staticmethod
    def info(key: str, value: str, user_gpt_context: UserGptContext) -> str:
        """Alias for addoptionalinfo\n
        /info <key> <value>"""
        return ChatGptCommands.addoptionalinfo(key, value, user_gpt_context=user_gpt_context)
