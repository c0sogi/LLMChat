from asyncio import gather
from enum import Enum
from typing import Optional

from app.models.chat_models import (
    ChatRoles,
    ResponseType,
    UserChatContext,
    command_response,
)
from app.models.llms import LLMModels
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.chat_rooms import delete_chat_room
from app.utils.chat.managers.cache import CacheManager
from app.utils.chat.managers.message import MessageManager
from app.utils.chat.managers.websocket import SendToWebsocket


class CoreCommands:
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
    async def deletechatroom(
        chat_room_id: str, buffer: BufferedUserContext
    ) -> None:
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
    async def retry(
        buffer: BufferedUserContext,
    ) -> tuple[Optional[str], ResponseType]:
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

    @classmethod
    async def model(
        cls, model: str, user_chat_context: UserChatContext
    ) -> str:
        """Alias for changemodel\n
        /model <model>"""
        return await cls.changemodel(model, user_chat_context)

    @staticmethod
    @command_response.send_message_and_stop
    async def changemodel(
        model: str, user_chat_context: UserChatContext
    ) -> str:
        """Change GPT model\n
        /changemodel <model>"""
        if model not in LLMModels.member_names:
            return f"Model must be one of {', '.join(LLMModels.member_names)}"
        llm_model: Enum = LLMModels.get_member(model)
        user_chat_context.llm_model = llm_model
        await CacheManager.update_profile_and_model(user_chat_context)
        return (
            f"Model changed to {model}. Actual model: {llm_model.value.name}"
        )

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
    @command_response.send_message_and_stop
    def info(
        cls, key: str, value: str, user_chat_context: UserChatContext
    ) -> str:
        """Alias for addoptionalinfo\n
        /info <key> <value>"""
        return cls.addoptionalinfo(
            key, value, user_chat_context=user_chat_context
        )
