import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from fastapi import WebSocket

from app.database.schemas.auth import Users
from app.models.chat_models import (
    ChatRoles,
    MessageHistory,
    UserChatContext,
    UserChatProfile,
    LLMModels,
)
from app.models.base_models import UserChatRoles


class ContextList:
    def __init__(
        self,
        initial_list: list[UserChatProfile],
        read_callback: Callable[[UserChatProfile], Awaitable[UserChatContext]],
    ):
        self.data: list[Any] = initial_list
        self.read_callback = read_callback

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    async def at(self, index: int) -> UserChatContext:
        if isinstance(self.data[index], UserChatProfile):
            self.data[index] = await self.read_callback(self.data[index])
        return self.data[index]

    def insert(self, index: int, value: Any) -> None:
        self.data.insert(index, value)

    def delete(self, index: int) -> None:
        del self.data[index]


@dataclass
class BufferedUserContext:
    user: Users
    websocket: WebSocket
    initialize_callback: Callable[[str], Awaitable[list[UserChatProfile]]]
    read_callback: Callable[[UserChatProfile], Awaitable[UserChatContext]]
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task_list: list[asyncio.Task[Any]] = field(default_factory=list)  # =
    last_user_message: Optional[str] = None
    optional_info: dict = field(default_factory=dict)
    is_stream_in_progress: asyncio.Event = field(default_factory=asyncio.Event)
    _sorted_ctxts: ContextList = field(init=False)
    _current_ctxt: UserChatContext = field(init=False)

    def __len__(self):
        return len(self._sorted_ctxts)

    async def init(self):
        self._sorted_ctxts = ContextList(
            initial_list=await self.initialize_callback(self.user.email),
            read_callback=self.read_callback,
        )
        self._current_ctxt = await self._sorted_ctxts.at(0)

    def insert_context(
        self, user_chat_context: UserChatContext, index: int = 0
    ) -> None:
        self._sorted_ctxts.insert(index=index, value=user_chat_context)

    def delete_context(self, index: int) -> None:
        self._sorted_ctxts.delete(index=index)

    def find_index_of_chatroom(self, chat_room_id: str) -> int | None:
        try:
            return self.sorted_chat_room_ids.index(chat_room_id)
        except ValueError:
            return None

    async def find_index_of_message_history(
        self,
        user_chat_context: UserChatContext,
        role: ChatRoles,
        message_history_uuid: str,
    ) -> int | None:  # =
        try:
            if role is ChatRoles.USER:
                message_histories = user_chat_context.user_message_histories
            elif role is ChatRoles.AI:
                message_histories = user_chat_context.ai_message_histories
            elif role is ChatRoles.SYSTEM:
                message_histories = user_chat_context.system_message_histories
            else:
                raise ValueError("role must be one of 'user', 'ai', 'system'")
            for message_history_index, message_history in enumerate(message_histories):
                if message_history.uuid == message_history_uuid:
                    return message_history_index
        except ValueError:
            return None

    async def change_context_to(self, index: int) -> None:
        self._current_ctxt = await self._sorted_ctxts.at(index)

    @property
    def current_user_chat_roles(self) -> UserChatRoles:
        return self._current_ctxt.llm_model.value.user_chat_roles

    @property
    def sorted_chat_room_ids(self) -> list[str]:
        return [
            context_or_profile.chat_room_id for context_or_profile in self._sorted_ctxts
        ]

    @property
    def sorted_chat_rooms(self) -> list[dict[str, str]]:
        return [
            {
                "chat_room_id": context_or_profile.chat_room_id,
                "chat_room_name": context_or_profile.chat_room_name,
            }
            for context_or_profile in self._sorted_ctxts
        ]

    @property
    def user_id(self) -> str:
        return self.user.email

    @property
    def current_chat_room_id(self) -> str:
        return self._current_ctxt.chat_room_id

    @property
    def current_llm_model(self) -> LLMModels:
        return self._current_ctxt.llm_model

    @property
    def current_chat_room_name(self) -> str:
        return self._current_ctxt.chat_room_name

    @current_chat_room_name.setter
    def current_chat_room_name(self, new_name: str) -> None:
        self._current_ctxt.user_chat_profile.chat_room_name = new_name

    @property
    def current_user_chat_context(self) -> UserChatContext:
        return self._current_ctxt

    @property
    def current_user_message_histories(self) -> list[MessageHistory]:
        return self._current_ctxt.user_message_histories

    @property
    def current_ai_message_histories(self) -> list[MessageHistory]:
        return self._current_ctxt.ai_message_histories

    @property
    def current_system_message_histories(self) -> list[MessageHistory]:
        return self._current_ctxt.system_message_histories

    @property
    def current_user_chat_profile(self) -> UserChatProfile:
        return self._current_ctxt.user_chat_profile
