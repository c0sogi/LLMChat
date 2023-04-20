from typing import Literal
from app.utils.chatgpt.chatgpt_context_manager import chatgpt_cache_manager
from app.viewmodels.gpt_models import MessageHistory, UserGptContext


class MessageManager:
    @staticmethod
    def add_message_history_safely(
        user_gpt_context: UserGptContext,
        content: str,
        role: Literal["user", "gpt", "system"],
    ) -> None:
        assert role in ("user", "gpt", "system")
        tokens: int = len(user_gpt_context.tokenize(content)) + 8
        message_history: MessageHistory = MessageHistory(
            role=getattr(user_gpt_context.user_gpt_profile, f"{role}_role"),
            content=content,
            tokens=tokens,
            is_user=True if role == "user" else False,
        )
        getattr(user_gpt_context, f"{role}_message_histories").append(message_history)
        setattr(
            user_gpt_context,
            f"{role}_message_tokens",
            getattr(user_gpt_context, f"{role}_message_tokens") + tokens,
        )
        num_of_deleted_histories: int = user_gpt_context.ensure_token_not_exceed()
        chatgpt_cache_manager.append_message_history(
            user_id=user_gpt_context.user_gpt_profile.user_id,
            role=role,
            message_history=message_history,
        )
        if num_of_deleted_histories > 0:
            for role in ("user", "gpt"):
                chatgpt_cache_manager.lpop_message_history(
                    user_id=user_gpt_context.user_gpt_profile.user_id,
                    role=role,
                    count=num_of_deleted_histories,
                )

    @staticmethod
    def rpop_message_history_safely(
        user_gpt_context: UserGptContext,
        role: Literal["user", "gpt"],
    ) -> MessageHistory | None:
        assert role in ("user", "gpt")
        try:
            message_history: MessageHistory = getattr(user_gpt_context, f"{role}_message_histories").pop()
        except IndexError:
            return None
        setattr(
            user_gpt_context,
            f"{role}_message_tokens",
            getattr(user_gpt_context, f"{role}_message_tokens") - message_history.tokens,
        )
        chatgpt_cache_manager.rpop_message_history(
            user_id=user_gpt_context.user_gpt_profile.user_id,
            role=role,
        )
        return message_history

    @staticmethod
    def set_message_history_safely(
        user_gpt_context: UserGptContext,
        new_content: str,
        role: Literal["user", "gpt", "system"],
        index: int,
    ) -> None:
        assert role in ("user", "gpt", "system")
        message_history_to_change: MessageHistory = getattr(user_gpt_context, f"{role}_message_histories")[index]
        new_tokens: int = len(user_gpt_context.tokenize(new_content)) + 8
        old_tokens: int = message_history_to_change.tokens
        message_history_to_change.content = new_content
        message_history_to_change.tokens = new_tokens
        setattr(
            user_gpt_context,
            f"{role}_message_tokens",
            getattr(user_gpt_context, f"{role}_message_tokens") + new_tokens - old_tokens,
        )
        num_of_deleted_histories: int = user_gpt_context.ensure_token_not_exceed()
        chatgpt_cache_manager.set_message_history(
            user_id=user_gpt_context.user_gpt_profile.user_id,
            role=role,
            message_history=message_history_to_change,
            index=index,
        )
        if num_of_deleted_histories > 0:
            for role in ("user", "gpt"):
                chatgpt_cache_manager.lpop_message_history(
                    user_id=user_gpt_context.user_gpt_profile.user_id,
                    role=role,
                    count=num_of_deleted_histories,
                )
