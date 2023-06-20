from asyncio import gather
from concurrent.futures import ProcessPoolExecutor
from inspect import Parameter, iscoroutinefunction, signature
from types import NoneType
from typing import Any, Callable, Optional, Tuple, Union, get_args, get_origin
from uuid import uuid4

from fastapi import WebSocket
from fastapi.concurrency import run_in_threadpool

from app.common.config import config
from app.common.constants import SystemPrompts
from app.errors.api_exceptions import InternalServerError
from app.models.chat_models import (
    ChatRoles,
    LLMModels,
    MessageHistory,
    ResponseType,
    UserChatContext,
    command_response,
)
from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.cache import CacheManager
from app.utils.chat.commands.browse import browse
from app.utils.chat.commands.vectorstore import query
from app.utils.chat.commands.summarize import summarize
from app.utils.chat.messages.handler import MessageHandler
from app.utils.chat.managers.message import MessageManager
from app.utils.chat.managers.vectorstore import VectorStoreManager
from app.utils.chat.managers.websocket import SendToWebsocket
from app.viewmodels.status import UserStatus


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


class ChatCommands:
    @classmethod
    def _find_callback_with_command(cls, command: str) -> Callable:
        return getattr(cls, command, cls.not_existing_callback)

    @staticmethod
    @command_response.send_message_and_stop
    def not_existing_callback() -> str:  # callback for not existing command
        return "Sorry, I don't know what you mean by..."

    @classmethod
    @command_response.send_message_and_stop
    def help(cls) -> str:
        docs: list[str] = [
            getattr(cls, callback_name).__doc__
            for callback_name in dir(cls)
            if not callback_name.startswith("_")
        ]
        return "\n\n".join([doc for doc in docs if doc is not None])

    @staticmethod
    @command_response.do_nothing
    async def deletechatroom(chat_room_id: str, buffer: BufferedUserContext) -> None:
        chat_room_id_before: str = buffer.current_chat_room_id
        delete_result: bool = await delete_chat_room(
            chat_room_id_to_delete=chat_room_id,
            buffer=buffer,
        )
        if buffer.current_chat_room_id == chat_room_id_before:
            await SendToWebsocket.init(
                buffer=buffer,
                send_previous_chats=False,
                send_chat_rooms=delete_result,
            )
        else:
            await SendToWebsocket.init(
                buffer=buffer,
                send_previous_chats=True,
                send_chat_rooms=delete_result,
            )

    @staticmethod
    @command_response.send_message_and_stop
    async def clear(user_chat_context: UserChatContext) -> str:
        """Clear user and ai message histories, and return the number of tokens removed\n
        /clear"""
        n_user_tokens: int = user_chat_context.user_message_tokens
        n_ai_tokens: int = user_chat_context.ai_message_tokens
        n_system_tokens: int = user_chat_context.system_message_tokens
        user_chat_context.user_message_histories.clear()
        user_chat_context.ai_message_histories.clear()
        user_chat_context.system_message_histories.clear()
        await CacheManager.delete_message_histories(
            user_id=user_chat_context.user_id,
            chat_room_id=user_chat_context.chat_room_id,
            roles=[ChatRoles.USER, ChatRoles.AI, ChatRoles.SYSTEM],
        )
        response: str = f"""## Total Token Removed: **{n_user_tokens + n_ai_tokens + n_system_tokens}**
- User: {n_user_tokens}
- AI: {n_ai_tokens}
- System: {n_system_tokens}"""
        return response  # return success message

    @staticmethod
    @command_response.send_message_and_stop
    def test(
        user_chat_context: UserChatContext,
    ) -> str:  # test command showing user_chat_context
        """Test command showing user_chat_context\n
        /test"""
        return str(user_chat_context)

    @staticmethod
    @command_response.send_message_and_stop
    async def reset(
        user_chat_context: UserChatContext,
    ) -> str:  # reset user_chat_context
        """Reset user_chat_context\n
        /reset"""
        user_chat_context.reset()
        if await CacheManager.reset_context(
            user_id=user_chat_context.user_id,
            chat_room_id=user_chat_context.chat_room_id,
        ):  # if reset success
            return "Context reset success"
        else:
            return "Context reset failed"

    @staticmethod
    @command_response.send_message_and_stop
    async def system(
        system_message: str, /, user_chat_context: UserChatContext
    ) -> str:  # add system message
        """Add system message\n
        /system <system_message>"""
        await MessageManager.add_message_history_safely(
            user_chat_context=user_chat_context,
            content=system_message,
            role=ChatRoles.SYSTEM,
        )
        return f"Added system message: {system_message}"  # return success message

    @staticmethod
    @command_response.send_message_and_stop
    async def settemperature(
        temp_to_change: float, user_chat_context: UserChatContext
    ) -> str:  # set temperature of ai
        """Set temperature of ai\n
        /settemperature <temp_to_change>"""
        try:
            assert 0 <= temp_to_change <= 1  # assert temperature is between 0 and 1
        except AssertionError:  # if temperature is not between 0 and 1
            return "Temperature must be between 0 and 1"
        else:
            previous_temperature: str = str(
                user_chat_context.user_chat_profile.temperature
            )
            user_chat_context.user_chat_profile.temperature = temp_to_change
            await CacheManager.update_profile_and_model(
                user_chat_context
            )  # update user_chat_context
            return f"I've changed temperature from {previous_temperature} to {temp_to_change}."  # return success msg

    @classmethod
    async def temp(
        cls, temp_to_change: float, user_chat_context: UserChatContext
    ) -> str:  # alias for settemperature
        """Alias for settemperature\n
        /temp <temp_to_change>"""
        return await cls.settemperature(temp_to_change, user_chat_context)

    @staticmethod
    @command_response.send_message_and_stop
    async def settopp(
        top_p_to_change: float, user_chat_context: UserChatContext
    ) -> str:  # set top_p of ai
        """Set top_p of ai\n
        /settopp <top_p_to_change>"""
        try:
            assert 0 <= top_p_to_change <= 1  # assert top_p is between 0 and 1
        except AssertionError:  # if top_p is not between 0 and 1
            return "Top_p must be between 0 and 1."  # return fail message
        else:
            previous_top_p: str = str(user_chat_context.user_chat_profile.top_p)
            user_chat_context.user_chat_profile.top_p = top_p_to_change  # set top_p
            await CacheManager.update_profile_and_model(
                user_chat_context
            )  # update user_chat_context
            return f"I've changed top_p from {previous_top_p} to {top_p_to_change}."  # return success message

    @classmethod
    async def topp(
        cls, top_p_to_change: float, user_chat_context: UserChatContext
    ) -> str:  # alias for settopp
        """Alias for settopp\n
        /topp <top_p_to_change>"""
        return await cls.settopp(top_p_to_change, user_chat_context)

    @staticmethod
    @command_response.send_message_and_stop
    async def poplastmessage(role: str, user_chat_context: UserChatContext) -> str:
        """Pop last message (user or system or ai)\n
        /poplastmessage <user|system|ai>"""
        try:
            actual_role: ChatRoles = ChatRoles.get_member(role)
        except ValueError:
            return "Role must be one of user, system, ai"  # return fail message
        last_message_history: MessageHistory | None = (
            await MessageManager.pop_message_history_safely(
                user_chat_context=user_chat_context,
                role=actual_role,
            )
        )  # type: ignore
        if last_message_history is None:  # if last_message_history is None
            return f"There is no {role} message to pop."  # return fail message
        return f"Pop {role} message: {last_message_history.content}"  # return success message

    @staticmethod
    @command_response.send_message_and_stop
    async def setlastmessage(
        role, new_message: str, /, user_chat_context: UserChatContext
    ) -> str:
        """Set last message (user or system or ai)\n
        /setlastmessage <user|system|ai> <new_message>"""
        try:
            actual_role: ChatRoles = ChatRoles.get_member(role)
        except ValueError:
            return "Role must be one of user, system, ai"  # return fail message
        if (
            await MessageManager.set_message_history_safely(
                user_chat_context=user_chat_context,
                role=actual_role,
                index=-1,
                new_content=new_message,
            )
            is None
        ):  # if set message history failed
            return f"There is no {role} message to set."
        return f"Set {role} message: {new_message}"  # return success message

    @classmethod
    async def pop(cls, role: str, user_chat_context: UserChatContext) -> str:
        """Alias for poplastmessage\n
        /pop <user|system|ai>"""
        return await cls.poplastmessage(role, user_chat_context)

    @classmethod
    async def set(
        cls, role, new_message: str, /, user_chat_context: UserChatContext
    ) -> str:
        """Alias for setlastmessage\n
        /set <user|system|ai> <new_message>"""
        return await cls.setlastmessage(role, new_message, user_chat_context)

    @staticmethod
    async def retry(buffer: BufferedUserContext) -> Tuple[str | None, ResponseType]:
        """Retry last message\n
        /retry"""
        if buffer.last_user_message is None:
            if (
                len(buffer.current_ai_message_histories)
                == len(buffer.current_user_message_histories)
                > 0
            ):
                await MessageManager.pop_message_history_safely(
                    user_chat_context=buffer.current_user_chat_context,
                    role=ChatRoles.AI,
                )
                return (None, ResponseType.HANDLE_AI)
            else:
                return (
                    "There is no message to retry.",
                    ResponseType.SEND_MESSAGE_AND_STOP,
                )
        if buffer.last_user_message.startswith("/"):
            changable = False
            for command in ("/browse", "/query"):
                if buffer.last_user_message.startswith(command):
                    changable = True
                    break
            if changable and (
                len(buffer.current_ai_message_histories)
                == len(buffer.current_user_message_histories)
                > 0
            ):
                await gather(
                    MessageManager.pop_message_history_safely(
                        user_chat_context=buffer.current_user_chat_context,
                        role=ChatRoles.USER,
                    ),
                    MessageManager.pop_message_history_safely(
                        user_chat_context=buffer.current_user_chat_context,
                        role=ChatRoles.AI,
                    ),
                )
            return (buffer.last_user_message, ResponseType.REPEAT_COMMAND)
        else:
            if (
                len(buffer.current_ai_message_histories)
                == len(buffer.current_user_message_histories)
                > 0
            ):
                await MessageManager.pop_message_history_safely(
                    user_chat_context=buffer.current_user_chat_context,
                    role=ChatRoles.AI,
                )
            return (None, ResponseType.HANDLE_AI)

    @staticmethod
    @command_response.send_message_and_stop
    def ping() -> str:
        """Ping! Pong!\n
        /ping"""
        return "pong"

    @staticmethod
    @command_response.send_message_and_stop
    async def codex(user_chat_context: UserChatContext) -> str:
        """Let GPT act as CODEX("COding DEsign eXpert")\n
        /codex"""
        system_message = SystemPrompts.CODEX
        await MessageManager.clear_message_history_safely(
            user_chat_context=user_chat_context, role=ChatRoles.SYSTEM
        )
        await MessageManager.add_message_history_safely(
            user_chat_context=user_chat_context,
            role=ChatRoles.SYSTEM,
            content=system_message,
        )
        return "CODEX mode ON"

    @staticmethod
    @command_response.send_message_and_stop
    async def redx(user_chat_context: UserChatContext) -> str:
        """Let GPT reduce your message as much as possible\n
        /redx"""
        system_message = SystemPrompts.REDEX
        await MessageManager.clear_message_history_safely(
            user_chat_context=user_chat_context, role=ChatRoles.SYSTEM
        )
        await MessageManager.add_message_history_safely(
            user_chat_context=user_chat_context,
            role=ChatRoles.SYSTEM,
            content=system_message,
        )
        return "REDX mode ON"

    @staticmethod
    @command_response.send_message_and_stop
    def echo(msg: str, /) -> str:
        """Echo your message\n
        /echo <msg>"""
        return msg

    @staticmethod
    @command_response.do_nothing
    async def sendtowebsocket(
        msg: str, /, websocket: WebSocket, user_chat_context: UserChatContext
    ) -> None:
        """Send all messages to websocket\n
        /sendtowebsocket"""
        await SendToWebsocket.message(
            websocket=websocket,
            msg=msg,
            chat_room_id=user_chat_context.chat_room_id,
        )

    @staticmethod
    @command_response.send_message_and_stop
    def codeblock(language, codes: str, /) -> str:
        """Send codeblock\n
        /codeblock <language> <codes>"""
        return f"\n```{language.lower()}\n" + codes + "\n```\n"

    @classmethod
    async def model(cls, model: str, user_chat_context: UserChatContext) -> str:
        """Alias for changemodel\n
        /model <model>"""
        return await cls.changemodel(model, user_chat_context)

    @staticmethod
    @command_response.send_message_and_stop
    async def changemodel(model: str, user_chat_context: UserChatContext) -> str:
        """Change GPT model\n
        /changemodel <model>"""
        if model not in LLMModels._member_names_:
            return f"Model must be one of {', '.join(LLMModels._member_names_)}"
        llm_model: LLMModels = LLMModels.get_member(model)
        user_chat_context.llm_model = llm_model
        await CacheManager.update_profile_and_model(user_chat_context)
        return f"Model changed to {model}. Actual model: {llm_model.value.name}"

    @staticmethod
    @command_response.send_message_and_stop
    def addoptionalinfo(
        key: str, value: str, user_chat_context: UserChatContext
    ) -> str:
        """Add optional info to buffer\n
        /addoptionalinfo <key> <value>"""
        user_chat_context.optional_info[key] = " ".join(value)
        return f"I've added {key}={value} to your optional info."

    @classmethod
    def info(cls, key: str, value: str, user_chat_context: UserChatContext) -> str:
        """Alias for addoptionalinfo\n
        /info <key> <value>"""
        return cls.addoptionalinfo(key, value, user_chat_context=user_chat_context)

    @staticmethod
    async def testchaining(
        chain_size: int, buffer: BufferedUserContext
    ) -> Tuple[str, ResponseType]:
        """Test chains of commands\n
        /testchaining <size_of_chain>"""
        if chain_size <= 0:
            return "Chaining Complete!", ResponseType.SEND_MESSAGE_AND_STOP
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=f"Current Chaining: {chain_size}",
            chat_room_id=buffer.current_chat_room_id,
        )
        return f"/testchaining {chain_size-1}", ResponseType.REPEAT_COMMAND

    @staticmethod
    async def query(
        user_query: str, /, buffer: BufferedUserContext, **kwargs
    ) -> Tuple[str | None, ResponseType]:
        """Query from redis vectorstore\n
        /query <query>"""
        return await query(user_query, buffer=buffer, **kwargs)

    @staticmethod
    @command_response.send_message_and_stop
    async def embed(text_to_embed: str, /, buffer: BufferedUserContext) -> str:
        """Embed the text and save its vectors in the redis vectorstore.\n
        /embed <text_to_embed>"""
        await VectorStoreManager.create_documents(
            text=text_to_embed, collection_name=buffer.user_id
        )
        return "\n```lottie-ok\n### Embedding successful!\n```\n"

    @staticmethod
    @command_response.send_message_and_stop
    async def share(text_to_embed: str, /) -> str:
        """Embed the text and save its vectors in the redis vectorstore. This index is shared for everyone.\n
        /share <text_to_embed>"""
        await VectorStoreManager.create_documents(
            text=text_to_embed, collection_name=config.shared_vectorestore_name
        )
        return "\n```lottie-ok\n### Embedding successful!\nThis data will be shared for everyone.\n```\n"

    @staticmethod
    @command_response.send_message_and_stop
    async def drop(buffer: BufferedUserContext) -> str:
        """Drop the index from the redis vectorstore.\n
        /drop"""
        dropped_index: list[str] = []
        if await VectorStoreManager.delete_collection(collection_name=buffer.user_id):
            dropped_index.append(buffer.user_id)
        if (
            buffer.user.status is UserStatus.admin
            and await VectorStoreManager.delete_collection(
                collection_name=config.shared_vectorestore_name,
            )
        ):
            dropped_index.append(config.shared_vectorestore_name)
        if not dropped_index:
            return "No index dropped."
        return f"Index dropped: {', '.join(dropped_index)}"

    @staticmethod
    @command_response.send_message_and_stop
    async def summarize(
        to_summarize: Optional[str], /, buffer: BufferedUserContext
    ) -> str:
        """Summarize the conversation\n
        /summarize"""
        return await summarize(to_summarize, buffer=buffer)

    @staticmethod
    @command_response.send_message_and_stop
    async def free() -> str:
        """Free the process pool executor\n
        /free"""
        shared = Shared()
        shared.process_pool_executor.shutdown(wait=True)
        shared.process_pool_executor = ProcessPoolExecutor()
        return "Process pool executor freed!"

    @staticmethod
    @command_response.send_message_and_stop
    async def browse_searx(query: str, /) -> str:
        """Search web for the query, with searxNG\n
        /browse_searx <query>"""
        return await Shared().searx.arun(query)

    @staticmethod
    async def browse(
        user_query: str, /, buffer: BufferedUserContext, **kwargs
    ) -> Tuple[str | None, ResponseType]:
        """Query LLM with duckduckgo browse results\n
        /browse <query>"""
        return await browse(user_query, buffer=buffer, **kwargs)


async def command_handler(
    callback_name: str,
    callback_args: list[str],
    translate: Optional[str],
    buffer: BufferedUserContext,
):
    callback_response, response_type = await _get_command_response(
        callback_name=callback_name,
        callback_args=callback_args,
        buffer=buffer,
        translate=translate,
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


async def _get_command_response(
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
        callback: Callable = ChatCommands._find_callback_with_command(callback_name)
    try:
        args_to_pass, kwargs_to_pass = _arguments_provider(
            func=callback,
            available_args=callback_args,
            available_annotated={
                UserChatContext: buffer.current_user_chat_context,
                WebSocket: buffer.websocket,
                BufferedUserContext: buffer,
            },
            available_kwargs=buffer.current_user_chat_context.optional_info | kwargs,
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
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=callback_response,
                chat_room_id=buffer.current_chat_room_id,
            )
            return None, ResponseType.DO_NOTHING
