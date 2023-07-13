from app.common.constants import SystemPrompts
from app.models.base_models import MessageHistory
from app.models.chat_models import ChatRoles, UserChatContext, command_response
from app.utils.chat.managers.message import MessageManager


class PromptCommands:
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
        return (
            f"Added system message: {system_message}"  # return success message
        )

    @staticmethod
    @command_response.send_message_and_stop
    async def pop(role: str, user_chat_context: UserChatContext) -> str:
        """Pop last message (user or system or ai)\n
        /pop <user|system|ai>"""
        try:
            actual_role: ChatRoles = ChatRoles.get_static_member(role)
        except ValueError:
            return (
                "Role must be one of user, system, ai"  # return fail message
            )
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
    async def set(
        role, new_message: str, /, user_chat_context: UserChatContext
    ) -> str:
        """Set last message (user or system or ai)\n
        /set <user|system|ai> <new_message>"""
        try:
            actual_role: ChatRoles = ChatRoles.get_static_member(role)
        except ValueError:
            return (
                "Role must be one of user, system, ai"  # return fail message
            )
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
