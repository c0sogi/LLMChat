import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import WebSocket

from app.database.schemas.auth import Users
from app.models.chat_models import MessageHistory, UserChatContext, UserChatProfile

if TYPE_CHECKING:
    from app.models.llms import LLMModels


@dataclass
class BufferedUserContext:
    user: Users
    websocket: WebSocket
    sorted_ctxts: list[UserChatContext]
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    _current_ctxt: UserChatContext = field(init=False)

    def __post_init__(self) -> None:
        self._current_ctxt = self.sorted_ctxts[0]

    def insert_context(self, user_chat_context: UserChatContext, index: int = 0) -> None:
        self.sorted_ctxts.insert(index, user_chat_context)

    def delete_context(self, index: int) -> None:
        del self.sorted_ctxts[index]

    def find_index_of_chatroom(self, chat_room_id: str) -> int | None:
        try:
            return self.sorted_chat_room_ids.index(chat_room_id)
        except ValueError:
            return None

    def change_context_to(self, index: int) -> None:
        self._current_ctxt = self.sorted_ctxts[index]

    @property
    def user_id(self) -> str:
        return self.user.email

    @property
    def buffer_size(self) -> int:
        return len(self.sorted_ctxts)

    @property
    def current_chat_room_id(self) -> str:
        return self._current_ctxt.chat_room_id

    @property
    def current_llm_model(self) -> "LLMModels":
        return self._current_ctxt.llm_model

    @property
    def current_chat_room_name(self) -> str:
        return self._current_ctxt.chat_room_name

    @current_chat_room_name.setter
    def current_chat_room_name(self, new_name: str) -> None:
        self._current_ctxt.user_chat_profile.chat_room_name = new_name

    @property
    def sorted_user_chat_contexts(self) -> list[UserChatContext]:
        return self.sorted_ctxts

    @property
    def sorted_chat_room_ids(self) -> list[str]:
        return [context.chat_room_id for context in self.sorted_ctxts]

    @property
    def sorted_chat_room_names(self) -> list[str]:
        return [context.chat_room_name for context in self.sorted_ctxts]

    @property
    def sorted_chat_rooms(self) -> list[dict[str, str]]:
        return [
            {"chat_room_id": context.chat_room_id, "chat_room_name": context.chat_room_name}
            for context in self.sorted_ctxts
        ]

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
