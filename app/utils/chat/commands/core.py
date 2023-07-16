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
    @command_response.do_nothing
    async def deletemessage(
        role: str, message_uuid: str, buffer: BufferedUserContext
    ) -> None:
        actual_role = ChatRoles.get_static_member(role)
        message_index = await buffer.find_index_of_message_history(
            user_chat_context=buffer.current_user_chat_context,
            message_history_uuid=message_uuid,
            role=actual_role,
        )
        if message_index is not None:
            await MessageManager.delete_message_history_safely(
                user_chat_context=buffer.current_user_chat_context,
                role=actual_role,
                index=message_index,
                update_cache=True,
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
        last_user_message = buffer.last_user_message
        last_user_message_history = (
            buffer.current_user_message_histories[-1]
            if buffer.current_user_message_histories
            else None
        )
        last_ai_message_history = (
            buffer.current_ai_message_histories[-1]
            if buffer.current_ai_message_histories
            else None
        )
        if (
            last_ai_message_history
            and last_user_message_history
            and last_ai_message_history.timestamp
            > last_user_message_history.timestamp
        ):
            # The last ai message history is newer than the last user message history
            # so we have to remove the last ai message history
            await MessageManager.pop_message_history_safely(
                user_chat_context=buffer.current_user_chat_context,
                role=ChatRoles.AI,
            )

        if last_user_message and last_user_message_history is None:
            # There is no user message history, but there is a user message
            if last_user_message.startswith("/"):
                # Try to retry the last user message (which is a command)
                return last_user_message, ResponseType.REPEAT_COMMAND
            # Try to retry the last user message (which is not a command)
            return last_user_message, ResponseType.HANDLE_BOTH
        elif not last_user_message and last_user_message_history is not None:
            # There is no user message, but there is a user message history
            return None, ResponseType.HANDLE_AI
        elif last_user_message and last_user_message_history is not None:
            # We have both user message and user message history
            # so we have to remove the last user message history
            # since we are going to retry the last user message
            # instead of the last user message history
            # last user message has higher priority than last user message history
            await MessageManager.pop_message_history_safely(
                user_chat_context=buffer.current_user_chat_context,
                role=ChatRoles.USER,
            )
            if last_user_message.startswith("/"):
                # Try to retry the last user message (which is a command)
                return last_user_message, ResponseType.REPEAT_COMMAND
            # Try to retry the last user message (which is not a command)
            return last_user_message, ResponseType.HANDLE_BOTH
        else:
            # Can't retry since there is no user input to retry
            return (
                "You have no message to retry",
                ResponseType.SEND_MESSAGE_AND_STOP,
            )

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
        key: str, value: str, buffer: BufferedUserContext
    ) -> str:
        """Add optional info to buffer\n
        /addoptionalinfo <key> <value>"""
        buffer.optional_info[key] = " ".join(value)
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
