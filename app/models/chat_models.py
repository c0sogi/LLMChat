from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Union
from uuid import uuid4

from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads
from app.common.config import DEFAULT_LLM_MODEL

from app.models.llms import LLMModels
from app.utils.date_utils import UTC
from app.viewmodels.base_models import UserChatRoles


class ChatRoles(str, Enum):
    AI = "assistant"
    SYSTEM = "system"
    USER = "user"

    @classmethod
    def get_name(cls, role: Union["ChatRoles", str]) -> str:
        if isinstance(role, cls):  # when role is member
            return role.name
        elif not isinstance(role, str):
            raise ValueError(f"Invalid role: {role}")
        elif role in cls._value2member_map_:  # when role is value
            return cls._value2member_map_[role].name
        elif role.upper() in cls._member_map_:  # when role is name
            return role
        else:
            raise ValueError(f"Invalid role: {role}")

    @classmethod
    def get_value(cls, role: Union["ChatRoles", str]) -> str:
        if isinstance(role, cls):  # when role is member
            return role.value
        elif not isinstance(role, str):
            raise ValueError(f"Invalid role: {role}")
        elif role in cls._value2member_map_:  # when role is value
            return role
        elif role.upper() in cls._member_map_:  # when role is name
            return cls._member_map_[role.upper()].value
        else:
            raise ValueError(f"Invalid role: {role}")

    @classmethod
    def get_member(cls, role: Union["ChatRoles", str]) -> "ChatRoles":
        if isinstance(role, cls):  # when role is member
            return role
        elif role in cls._value2member_map_:  # when role is value
            return cls._value2member_map_[role]  # type: ignore
        elif not isinstance(role, str):
            raise ValueError(f"Invalid role: {role}")
        elif role.upper() in cls._member_map_:  # when role is name
            return cls._member_map_[role.upper()]  # type: ignore
        else:
            raise ValueError(f"Invalid role: {role}")


@dataclass
class MessageHistory:
    role: str
    content: str
    tokens: int
    is_user: bool
    timestamp: int = field(default_factory=UTC.timestamp)
    uuid: str = field(default_factory=lambda: uuid4().hex)
    model_name: str | None = None

    def __repr__(self) -> str:
        return f'<{self.role} uuid="{self.uuid}" date="{self.datetime}Z" tokens="{self.tokens}">{self.content}</>'

    @property
    def datetime(self) -> datetime:
        return UTC.timestamp_to_datetime(self.timestamp)


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
    llm_model: LLMModels
    user_message_histories: list[MessageHistory] = field(default_factory=list)
    ai_message_histories: list[MessageHistory] = field(default_factory=list)
    system_message_histories: list[MessageHistory] = field(default_factory=list)

    optional_info: dict = field(default_factory=dict)

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
            llm_model=LLMModels._member_map_[stored["llm_model"].replace(".", "_").replace("-", "_")],  # type: ignore
            user_message_histories=[MessageHistory(**m) for m in stored["user_message_histories"]],
            ai_message_histories=[MessageHistory(**m) for m in stored["ai_message_histories"]],
            system_message_histories=[MessageHistory(**m) for m in stored["system_message_histories"]],
        )

    def json(self) -> dict:
        return {
            "user_chat_profile": asdict(self.user_chat_profile),
            "llm_model": self.llm_model.name,
            "user_message_histories": [m.__dict__ for m in self.user_message_histories],
            "ai_message_histories": [m.__dict__ for m in self.ai_message_histories],
            "system_message_histories": [m.__dict__ for m in self.system_message_histories],
        }

    def to_stringified_json(self) -> str:
        return orjson_dumps(self.json()).decode("utf-8")

    def tokenize(self, message: str) -> list[int]:
        return self.llm_model.value.tokenizer.encode(message)

    def get_tokens_of(self, message: str) -> int:
        return len(self.tokenize(message))

    @property
    def left_tokens(self) -> int:
        return (
            self.llm_model.value.max_total_tokens
            - self.total_tokens
            - self.llm_model.value.token_margin
            - int(getattr(self.llm_model.value, "description_tokens", 0))
        )

    @property
    def total_tokens(self) -> int:
        return self.user_message_tokens + self.ai_message_tokens + self.system_message_tokens

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
        llm_model: LLMModels = self.llm_model
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
        llm_model: LLMModels = getattr(
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

    def ensure_token_not_exceed(self, extra_token_margin: int = 0) -> int:
        deleted_histories: int = 0
        while (
            len(self.user_message_histories) > 0
            and len(self.ai_message_histories) > 0
            and self.left_tokens < extra_token_margin
        ):
            deleted_histories += 1
            self.user_message_histories.pop(0)
            self.ai_message_histories.pop(0)
        return deleted_histories

    def clear_tokens(self, tokens_to_remove: int) -> int:
        deleted_histories: int = 0
        removed_tokens: int = 0
        while (
            len(self.user_message_histories) > 0
            and len(self.ai_message_histories) > 0
            and removed_tokens < tokens_to_remove
        ):
            deleted_histories += 1
            self.user_message_histories.pop(0)
            self.ai_message_histories.pop(0)
        return deleted_histories
