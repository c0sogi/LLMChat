from typing import Optional
from uuid import uuid4

from app.models.chat_models import ChatRoles, MessageHistory, UserChatContext

from .cache import CacheManager


class MessageManager:
    @staticmethod
    async def add_message_history_safely(
        user_chat_context: UserChatContext,
        content: str,
        role: ChatRoles,
        custom_role: Optional[str] = None,
        update_cache: bool = True,
        calculated_tokens_to_use: Optional[int] = None,
        uuid: Optional[str] = None,
    ) -> None:
        if role is ChatRoles.AI:
            user_chat_role = user_chat_context.user_chat_roles.ai
            actual_role: str = ChatRoles.AI.value
            histories_to_update: list[
                MessageHistory
            ] = user_chat_context.ai_message_histories
        elif role is ChatRoles.USER:
            user_chat_role = user_chat_context.user_chat_roles.user
            actual_role = ChatRoles.USER.value
            histories_to_update = user_chat_context.user_message_histories
        elif role is ChatRoles.SYSTEM:
            user_chat_role = user_chat_context.user_chat_roles.system
            actual_role = ChatRoles.SYSTEM.value
            histories_to_update = user_chat_context.system_message_histories
        else:
            raise ValueError(f"Invalid role: {role}")
        message_history: MessageHistory = MessageHistory(
            role=user_chat_role if custom_role is None else custom_role,
            content=content,
            tokens=user_chat_context.get_tokens_of(content)
            + user_chat_context.llm_model.value.token_margin
            if calculated_tokens_to_use is None
            else calculated_tokens_to_use
            + user_chat_context.llm_model.value.token_margin,
            actual_role=actual_role,
            model_name=user_chat_context.llm_model.value.name
            if role is ChatRoles.AI
            else None,
            uuid=uuid if uuid is not None else uuid4().hex,
        )
        histories_to_update.append(message_history)
        if update_cache:
            await CacheManager.append_message_history(
                user_id=user_chat_context.user_id,
                chat_room_id=user_chat_context.chat_room_id,
                role=role,
                message_history=message_history,
            )

    @staticmethod
    async def pop_message_history_safely(
        user_chat_context: UserChatContext,
        role: ChatRoles,
        count: int | None = None,
        rpop: bool = True,  # if False, lpop
    ) -> list[MessageHistory] | MessageHistory | None:
        if role is ChatRoles.AI:
            histories_to_pop: list[
                MessageHistory
            ] = user_chat_context.ai_message_histories
        elif role is ChatRoles.USER:
            histories_to_pop = user_chat_context.user_message_histories
        elif role is ChatRoles.SYSTEM:
            histories_to_pop = user_chat_context.system_message_histories
        else:
            raise ValueError(f"Invalid role: {role}")
        if count is None or count == 1:
            try:
                result = (
                    histories_to_pop.pop() if rpop else histories_to_pop.pop(0)
                )
            except IndexError:
                return None
        else:
            result = []
            try:
                for _ in range(count):
                    result.append(
                        histories_to_pop.pop()
                        if rpop
                        else histories_to_pop.pop(0)
                    )
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
        role: ChatRoles,
        index: int,
        new_content: Optional[str] = None,
        summarized_content: Optional[str] = None,  # =
        update_cache: bool = True,
    ) -> None:
        try:
            if role is ChatRoles.AI:
                histories_to_change: MessageHistory = (
                    user_chat_context.ai_message_histories[index]
                )
            elif role is ChatRoles.USER:
                histories_to_change = user_chat_context.user_message_histories[
                    index
                ]
            elif role is ChatRoles.SYSTEM:
                histories_to_change = (
                    user_chat_context.system_message_histories[index]
                )
            else:
                raise ValueError(f"Invalid role: {role}")
        except IndexError:
            return None
        if new_content is not None:
            histories_to_change.content = new_content
            histories_to_change.tokens = user_chat_context.get_tokens_of(
                new_content
            )
        if summarized_content is not None:
            histories_to_change.summarized = summarized_content
            histories_to_change.summarized_tokens = (
                user_chat_context.get_tokens_of(summarized_content)
                + user_chat_context.llm_model.value.token_margin
            )
        if update_cache:
            await CacheManager.set_message_history(
                user_id=user_chat_context.user_id,
                chat_room_id=user_chat_context.chat_room_id,
                role=role,
                message_history=histories_to_change,
                index=index,
            )

    @classmethod
    async def delete_message_history_safely(
        cls,
        user_chat_context: UserChatContext,
        role: ChatRoles,
        index: int,
        update_cache: bool = True,
    ) -> None:
        try:
            if role is ChatRoles.AI:
                user_chat_context.ai_message_histories.pop(index)
            elif role is ChatRoles.USER:
                user_chat_context.user_message_histories.pop(index)
            elif role is ChatRoles.SYSTEM:
                user_chat_context.system_message_histories.pop(index)
            else:
                raise ValueError(f"Invalid role: {role}")
        except IndexError:
            return None
        if update_cache:
            await CacheManager.delete_one_message_history(
                user_id=user_chat_context.user_id,
                chat_room_id=user_chat_context.chat_room_id,
                role=role,
                index=index,
            )

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
            n_deleted = len(user_chat_context.user_message_histories)
            user_chat_context.user_message_histories = []
        elif role is ChatRoles.SYSTEM:
            n_deleted = len(user_chat_context.system_message_histories)
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
