from uuid import uuid4

from app.models.chat_models import UserChatContext

from .buffer import BufferedUserContext
from .managers.cache import CacheManager


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
