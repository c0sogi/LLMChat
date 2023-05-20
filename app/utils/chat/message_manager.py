from asyncio import gather
from app.utils.chat.cache_manager import CacheManager
from app.models.chat_models import ChatRoles, MessageHistory, UserChatContext


class MessageManager:
    @staticmethod
    async def add_message_history_safely(
        user_chat_context: UserChatContext,
        content: str,
        role: ChatRoles,
        calculated_tokens_to_use: int | None = None,
        extra_token_margin: int = 0,
        update_cache: bool = True,
    ) -> int:
        if role is ChatRoles.AI:
            user_defined_role: str = user_chat_context.user_chat_profile.ai_role
            histories_to_update: list[MessageHistory] = user_chat_context.ai_message_histories
        elif role is ChatRoles.USER:
            user_defined_role: str = user_chat_context.user_chat_profile.user_role
            histories_to_update: list[MessageHistory] = user_chat_context.user_message_histories
        elif role is ChatRoles.SYSTEM:
            user_defined_role: str = user_chat_context.user_chat_profile.system_role
            histories_to_update: list[MessageHistory] = user_chat_context.system_message_histories
        else:
            raise ValueError(f"Invalid role: {role}")
        message_history: MessageHistory = MessageHistory(
            role=user_defined_role,
            content=content,
            tokens=user_chat_context.get_tokens_of(content) + user_chat_context.llm_model.value.token_margin
            if calculated_tokens_to_use is None
            else calculated_tokens_to_use + user_chat_context.llm_model.value.token_margin,
            is_user=True if role is ChatRoles.USER else False,
            model_name=user_chat_context.llm_model.value.name if role is ChatRoles.AI else None,
        )
        histories_to_update.append(message_history)
        num_of_deleted_histories: int = user_chat_context.ensure_token_not_exceed(extra_token_margin=extra_token_margin)
        if update_cache:
            await CacheManager.append_message_history(
                user_id=user_chat_context.user_id,
                chat_room_id=user_chat_context.chat_room_id,
                role=role,
                message_history=message_history,
            )

            if num_of_deleted_histories > 0:
                await gather(
                    CacheManager.lpop_message_history(
                        user_id=user_chat_context.user_id,
                        chat_room_id=user_chat_context.chat_room_id,
                        role=ChatRoles.AI,
                        count=num_of_deleted_histories,
                    ),
                    CacheManager.lpop_message_history(
                        user_id=user_chat_context.user_id,
                        chat_room_id=user_chat_context.chat_room_id,
                        role=ChatRoles.USER,
                        count=num_of_deleted_histories,
                    ),
                )
        return num_of_deleted_histories

    @staticmethod
    async def pop_message_history_safely(
        user_chat_context: UserChatContext,
        role: ChatRoles,
        count: int | None = None,
        rpop: bool = True,  # if False, lpop
    ) -> list[MessageHistory] | MessageHistory | None:
        if role is ChatRoles.AI:
            histories_to_pop: list[MessageHistory] = user_chat_context.ai_message_histories
        elif role is ChatRoles.USER:
            histories_to_pop: list[MessageHistory] = user_chat_context.user_message_histories
        elif role is ChatRoles.SYSTEM:
            histories_to_pop: list[MessageHistory] = user_chat_context.system_message_histories
        else:
            raise ValueError(f"Invalid role: {role}")
        if count is None or count == 1:
            try:
                result = histories_to_pop.pop() if rpop else histories_to_pop.pop(0)
            except IndexError:
                return None
        else:
            result = []
            try:
                for _ in range(count):
                    result.append(histories_to_pop.pop() if rpop else histories_to_pop.pop(0))
            except IndexError:
                pass
            finally:
                count = len(result)
        if rpop:
            await CacheManager.rpop_message_history(
                user_id=user_chat_context.user_id,
                chat_room_id=user_chat_context.chat_room_id,
                role=role,
                count=count,
            )
        else:
            await CacheManager.lpop_message_history(
                user_id=user_chat_context.user_id,
                chat_room_id=user_chat_context.chat_room_id,
                role=role,
                count=count,
            )
        return result

    @staticmethod
    async def set_message_history_safely(
        user_chat_context: UserChatContext,
        new_content: str,
        role: ChatRoles,
        index: int,
        extra_token_margin: int = 0,
        update_cache: bool = True,
    ) -> int | None:
        try:
            if role is ChatRoles.AI:
                histories_to_change: MessageHistory = user_chat_context.ai_message_histories[index]
            elif role is ChatRoles.USER:
                histories_to_change: MessageHistory = user_chat_context.user_message_histories[index]
            elif role is ChatRoles.SYSTEM:
                histories_to_change: MessageHistory = user_chat_context.system_message_histories[index]
            else:
                raise ValueError(f"Invalid role: {role}")
        except IndexError:
            return None
        histories_to_change.content = new_content
        histories_to_change.tokens = user_chat_context.get_tokens_of(new_content)
        num_of_deleted_histories: int = user_chat_context.ensure_token_not_exceed(extra_token_margin=extra_token_margin)
        if update_cache:
            await CacheManager.set_message_history(
                user_id=user_chat_context.user_id,
                chat_room_id=user_chat_context.chat_room_id,
                role=role,
                message_history=histories_to_change,
                index=index,
            )
            if num_of_deleted_histories > 0:
                await gather(
                    CacheManager.lpop_message_history(
                        user_id=user_chat_context.user_id,
                        chat_room_id=user_chat_context.chat_room_id,
                        role=ChatRoles.AI,
                        count=num_of_deleted_histories,
                    ),
                    CacheManager.lpop_message_history(
                        user_id=user_chat_context.user_id,
                        chat_room_id=user_chat_context.chat_room_id,
                        role=ChatRoles.USER,
                        count=num_of_deleted_histories,
                    ),
                )
        return num_of_deleted_histories

    @staticmethod
    async def clear_message_history_safely(
        user_chat_context: UserChatContext,
        role: ChatRoles,
        update_cache: bool = True,
    ) -> int:
        if role is ChatRoles.AI:
            n_deleted: int = len(user_chat_context.ai_message_histories)
            user_chat_context.ai_message_histories = []
        elif role is ChatRoles.USER:
            n_deleted: int = len(user_chat_context.user_message_histories)
            user_chat_context.user_message_histories = []
        elif role is ChatRoles.SYSTEM:
            n_deleted: int = len(user_chat_context.system_message_histories)
            user_chat_context.system_message_histories = []
        else:
            raise ValueError(f"Invalid role: {role}")
        if update_cache:
            await CacheManager.delete_message_history(
                user_id=user_chat_context.user_id,
                chat_room_id=user_chat_context.chat_room_id,
                role=role,
            )
        return n_deleted
