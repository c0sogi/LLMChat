from dataclasses import dataclass, field

from fastapi import WebSocket

from app.models.gpt_models import UserGptContext


@dataclass
class BufferedUserContext:
    user_id: str
    websocket: WebSocket
    sorted_ctxts: list[UserGptContext]
    _current_ctxt: UserGptContext = field(init=False)

    def __post_init__(self) -> None:
        self._current_ctxt = self.sorted_ctxts[0]

    def insert_context(self, user_gpt_context: UserGptContext, index: int = 0) -> None:
        self.sorted_ctxts.insert(index, user_gpt_context)

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
    def buffer_size(self) -> int:
        return len(self.sorted_ctxts)

    @property
    def current_chat_room_id(self) -> str:
        return self.current_user_gpt_context.chat_room_id

    @property
    def sorted_user_gpt_contexts(self) -> list[UserGptContext]:
        return self.sorted_ctxts

    @property
    def sorted_chat_room_ids(self) -> list[str]:
        return [context.chat_room_id for context in self.sorted_ctxts]

    @property
    def current_user_gpt_context(self) -> UserGptContext:
        return self._current_ctxt
