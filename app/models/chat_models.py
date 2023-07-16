from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, Awaitable, Callable, Tuple
from uuid import uuid4

from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads
from app.common.config import DEFAULT_LLM_MODEL
from app.mixins.enum import EnumMixin
from app.utils.date_utils import UTC

from .base_models import MessageHistory, UserChatRoles
from .llms import LLMModels


class ChatRoles(EnumMixin):
    AI = "assistant"
    SYSTEM = "system"
    USER = "user"


@dataclass
class UserChatProfile:
    user_id: str
    chat_room_id: str = field(default_factory=lambda: uuid4().hex)
    chat_room_name: str = field(default_factory=lambda: UTC.now_isoformat())
    created_at: int = field(default_factory=lambda: UTC.timestamp())
    temperature: float = 0.9
    top_p: float = 1.0
    presence_penalty: float = 0
    frequency_penalty: float = 1.1


@dataclass
class UserChatContext:
    user_chat_profile: UserChatProfile
    llm_model: Enum
    user_message_histories: list[MessageHistory] = field(default_factory=list)
    ai_message_histories: list[MessageHistory] = field(default_factory=list)
    system_message_histories: list[MessageHistory] = field(
        default_factory=list
    )

    @property
    def user_message_tokens(self) -> int:
        return sum([m.tokens for m in self.user_message_histories])

    @property
    def ai_message_tokens(self) -> int:
        return sum([m.tokens for m in self.ai_message_histories])

    @property
    def system_message_tokens(self) -> int:
        return sum([m.tokens for m in self.system_message_histories])

    @classmethod
    def parse_stringified_json(cls, stred_json: str) -> "UserChatContext":
        stored: dict = orjson_loads(stred_json)
        return cls(
            user_chat_profile=UserChatProfile(**stored["user_chat_profile"]),
            llm_model=LLMModels.get_member(
                stored["llm_model"].replace(".", "_").replace("-", "_"),
            ),
            user_message_histories=[
                MessageHistory(**m) for m in stored["user_message_histories"]
            ],
            ai_message_histories=[
                MessageHistory(**m) for m in stored["ai_message_histories"]
            ],
            system_message_histories=[
                MessageHistory(**m) for m in stored["system_message_histories"]
            ],
        )

    def json(self) -> dict:
        return {
            "user_chat_profile": asdict(self.user_chat_profile),
            "llm_model": self.llm_model.name,
            "user_message_histories": [
                m.__dict__ for m in self.user_message_histories
            ],
            "ai_message_histories": [
                m.__dict__ for m in self.ai_message_histories
            ],
            "system_message_histories": [
                m.__dict__ for m in self.system_message_histories
            ],
        }

    def to_stringified_json(self) -> str:
        return orjson_dumps(self.json()).decode("utf-8")

    def tokenize(self, message: str) -> list[int]:
        try:
            return self.llm_model.value.tokenizer.encode(message)
        except Exception:
            return []

    def get_tokens_of(self, message: str) -> int:
        return len(self.tokenize(message))

    @property
    def left_tokens(self) -> int:
        return (
            self.llm_model.value.max_total_tokens
            - self.total_tokens
            - self.llm_model.value.token_margin
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.user_message_tokens
            + self.ai_message_tokens
            + self.system_message_tokens
            + self.llm_model.value.prefix_tokens
            + self.llm_model.value.suffix_tokens
        )

    @property
    def token_per_request(self) -> int:
        return self.llm_model.value.max_tokens_per_request

    @property
    def user_chat_roles(self) -> UserChatRoles:
        return self.llm_model.value.user_chat_roles

    @property
    def user_id(self) -> str:
        return self.user_chat_profile.user_id

    @property
    def chat_room_id(self) -> str:
        return self.user_chat_profile.chat_room_id

    @property
    def chat_room_name(self) -> str:
        return self.user_chat_profile.chat_room_name

    def __repr__(self) -> str:
        llm_model: Enum = self.llm_model
        time_string: str = datetime.strptime(
            str(self.user_chat_profile.created_at),
            "%Y%m%d%H%M%S",
        ).strftime("%Y-%m-%d %H:%M:%S")
        return f"""# User Info
- Your ID: `{self.user_id}`
- This chatroom ID: `{self.chat_room_id}`
- Your profile created at: `{time_string}`
- User role: `{self.user_chat_roles.user}`
- AI role: `{self.user_chat_roles.ai}`
- System role: `{self.user_chat_roles.system}`

# LLM Info
- Model Name: `{llm_model.name}`
- Actual Model Name: `{llm_model.value.name}`
- Temperature: `{self.user_chat_profile.temperature}`
- Top P: `{self.user_chat_profile.top_p}`
- Presence Penalty: `{self.user_chat_profile.presence_penalty}`
- Frequency Penalty: `{self.user_chat_profile.frequency_penalty}`

# Token Info
- Maximum Token Limit: `{llm_model.value.max_total_tokens}`
- User Token Consumed: `{self.user_message_tokens}`
- AI Token Consumed: `{self.ai_message_tokens}`
- System Token Consumed: `{self.system_message_tokens}`
- Total Token Consumed: `{self.total_tokens}`
- Remaining Tokens: `{self.left_tokens}`

# Message Histories
- User Message_Histories={self.user_message_histories}

- AI Message Histories={self.ai_message_histories}

- System Message Histories={self.system_message_histories}
"""

    @classmethod
    def construct_default(
        cls,
        user_id: str,
        chat_room_id: str,
        llm_model: Enum = getattr(
            LLMModels,
            DEFAULT_LLM_MODEL,
            LLMModels.gpt_3_5_turbo,
        ),
    ):
        return cls(
            user_chat_profile=UserChatProfile(
                user_id=user_id,
                chat_room_id=chat_room_id,
            ),
            llm_model=llm_model,
        )

    def reset(self):
        for k, v in self.construct_default(
            self.user_id,
            self.chat_room_id,
            self.llm_model,
        ).__dict__.items():
            setattr(self, k, v)

    def copy_from(self, user_chat_context: "UserChatContext") -> None:
        for k, v in user_chat_context.__dict__.items():
            setattr(self, k, v)


class ResponseType(str, Enum):
    SEND_MESSAGE_AND_STOP = "send_message_and_stop"
    HANDLE_USER = "handle_user"
    HANDLE_AI = "handle_ai"
    HANDLE_BOTH = "handle_both"
    DO_NOTHING = "do_nothing"
    REPEAT_COMMAND = "repeat_command"


class command_response:
    @staticmethod
    def _wrapper(
        enum_type: ResponseType,
    ) -> Callable[..., Callable]:
        def decorator(
            func: Callable,
        ) -> (
            Callable[
                ...,
                Tuple[Any, ResponseType],
            ]
            | Callable[..., Awaitable[Tuple[Any, ResponseType]]]
        ):
            @wraps(func)
            def sync_wrapper(
                *args: Any, **kwargs: Any
            ) -> Tuple[Any, ResponseType]:
                result = func(*args, **kwargs)
                return (result, enum_type)

            @wraps(func)
            async def async_wrapper(
                *args: Any, **kwargs: Any
            ) -> Tuple[Any, ResponseType]:
                result = await func(*args, **kwargs)
                return (result, enum_type)

            return async_wrapper if iscoroutinefunction(func) else sync_wrapper

        return decorator

    send_message_and_stop = _wrapper(ResponseType.SEND_MESSAGE_AND_STOP)
    handle_user = _wrapper(ResponseType.HANDLE_USER)
    handle_ai = _wrapper(ResponseType.HANDLE_AI)
    handle_both = _wrapper(ResponseType.HANDLE_BOTH)
    do_nothing = _wrapper(ResponseType.DO_NOTHING)
    repeat_command = _wrapper(ResponseType.REPEAT_COMMAND)
