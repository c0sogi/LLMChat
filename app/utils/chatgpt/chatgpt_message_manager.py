import asyncio
from app.utils.chatgpt.chatgpt_cache_manager import ChatGptCacheManager
from app.models.gpt_models import GptRoles, MessageHistory, UserGptContext


class MessageManager:
    @staticmethod
    async def add_message_history_safely(
        user_gpt_context: UserGptContext,
        content: str,
        role: GptRoles | str,
        calculated_tokens_to_use: int | None = None,
        extra_token_margin: int = 0,
    ) -> None:
        role = GptRoles.get_name(role).lower()
        if calculated_tokens_to_use is None:
            tokens: int = user_gpt_context.get_tokens_of(content) + user_gpt_context.gpt_model.value.token_margin
        else:
            tokens: int = calculated_tokens_to_use + user_gpt_context.gpt_model.value.token_margin
        message_history: MessageHistory = MessageHistory(
            role=getattr(user_gpt_context.user_gpt_profile, f"{role}_role"),
            content=content,
            tokens=tokens,
            is_user=True if role == GptRoles.USER.name.lower() else False,
            model_name=user_gpt_context.gpt_model.value.name if GptRoles.get_member(role) is GptRoles.GPT else None,
        )
        getattr(user_gpt_context, f"{role}_message_histories").append(message_history)
        setattr(
            user_gpt_context,
            f"{role}_message_tokens",
            getattr(user_gpt_context, f"{role}_message_tokens") + tokens,
        )
        num_of_deleted_histories: int = user_gpt_context.ensure_token_not_exceed(extra_token_margin=extra_token_margin)
        await ChatGptCacheManager.append_message_history(
            user_id=user_gpt_context.user_id,
            chat_room_id=user_gpt_context.chat_room_id,
            role=role,
            message_history=message_history,
        )

        if num_of_deleted_histories > 0:
            asyncio.gather(
                ChatGptCacheManager.lpop_message_history(
                    user_id=user_gpt_context.user_id,
                    chat_room_id=user_gpt_context.chat_room_id,
                    role=GptRoles.GPT,
                    count=num_of_deleted_histories,
                ),
                ChatGptCacheManager.lpop_message_history(
                    user_id=user_gpt_context.user_id,
                    chat_room_id=user_gpt_context.chat_room_id,
                    role=GptRoles.USER,
                    count=num_of_deleted_histories,
                ),
            )

    @staticmethod
    async def pop_message_history_safely(
        user_gpt_context: UserGptContext,
        role: GptRoles | str,
        count: int | None = None,
        rpop: bool = True,  # if False, lpop
    ) -> MessageHistory | None:
        role = GptRoles.get_name(role).lower()
        try:
            message_history: MessageHistory = getattr(user_gpt_context, f"{role}_message_histories").pop()
        except IndexError:
            return None
        setattr(
            user_gpt_context,
            f"{role}_message_tokens",
            getattr(user_gpt_context, f"{role}_message_tokens") - message_history.tokens,
        )
        if rpop:
            await ChatGptCacheManager.rpop_message_history(
                user_id=user_gpt_context.user_id,
                chat_room_id=user_gpt_context.chat_room_id,
                role=role,
                count=count,
            )
        else:
            await ChatGptCacheManager.lpop_message_history(
                user_id=user_gpt_context.user_id,
                chat_room_id=user_gpt_context.chat_room_id,
                role=role,
                count=count,
            )
        return message_history

    @staticmethod
    async def set_message_history_safely(
        user_gpt_context: UserGptContext,
        new_content: str,
        role: GptRoles | str,
        index: int,
    ) -> None:
        role = GptRoles.get_name(role).lower()
        message_history_to_change: MessageHistory = getattr(user_gpt_context, f"{role}_message_histories")[index]
        new_tokens: int = user_gpt_context.get_tokens_of(new_content) + 8
        old_tokens: int = message_history_to_change.tokens
        message_history_to_change.content = new_content
        message_history_to_change.tokens = new_tokens
        setattr(
            user_gpt_context,
            f"{role}_message_tokens",
            getattr(user_gpt_context, f"{role}_message_tokens") + new_tokens - old_tokens,
        )
        num_of_deleted_histories: int = user_gpt_context.ensure_token_not_exceed()
        await ChatGptCacheManager.set_message_history(
            user_id=user_gpt_context.user_id,
            chat_room_id=user_gpt_context.chat_room_id,
            role=role,
            message_history=message_history_to_change,
            index=index,
        )
        if num_of_deleted_histories > 0:
            for role in (GptRoles.GPT, GptRoles.USER):
                await ChatGptCacheManager.lpop_message_history(
                    user_id=user_gpt_context.user_id,
                    chat_room_id=user_gpt_context.chat_room_id,
                    role=role,
                    count=num_of_deleted_histories,
                )

    @staticmethod
    async def clear_message_history_safely(
        user_gpt_context: UserGptContext,
        role: GptRoles | str,
    ) -> None:
        role = GptRoles.get_name(role).lower()
        setattr(user_gpt_context, f"{role}_message_histories", [])
        setattr(user_gpt_context, f"{role}_message_tokens", 0)
        await ChatGptCacheManager.delete_message_history(
            user_id=user_gpt_context.user_id,
            chat_room_id=user_gpt_context.chat_room_id,
            role=role,
        )
