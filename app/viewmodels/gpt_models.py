from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from uuid import uuid4
from typing import Type, Union

import orjson
import tiktoken

from app.utils.date_utils import UTC


class GptRoles(str, Enum):
    GPT = "assistant"
    SYSTEM = "system"
    USER = "user"

    @classmethod
    def get_name(cls, role: Union["GptRoles", str]) -> str:
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
    def get_value(cls, role: Union["GptRoles", str]) -> str:
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
    def get_member(cls, role: Union["GptRoles", str]) -> Enum:
        if isinstance(role, cls):  # when role is member
            return role
        elif role in cls._value2member_map_:  # when role is value
            return cls._value2member_map_[role]
        elif not isinstance(role, str):
            raise ValueError(f"Invalid role: {role}")
        elif role.upper() in cls._member_map_:  # when role is name
            return cls._member_map_[role.upper()]
        else:
            raise ValueError(f"Invalid role: {role}")


@dataclass
class LLMModel:
    name: str  # model name
    max_total_tokens: int
    max_tokens_per_request: int
    token_margin: int = 8
    api_url: str = "https://api.openai.com/v1/chat/completions"
    tokenizer: tiktoken.core.Encoding | None = None

    def __post_init__(self):
        if self.tokenizer is None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.name)
            except KeyError:
                raise ValueError(f"Invalid model name: {self.name}")
        assert isinstance(self.tokenizer, tiktoken.core.Encoding), "Invalid tokenizer"


class OpenAIModels(Enum):  # gpt models for openai api
    gpt_3_5_turbo = LLMModel(
        name="gpt-3.5-turbo",
        api_url="https://api.openai.com/v1/chat/completions",
        max_total_tokens=4096,
        max_tokens_per_request=4096,
        token_margin=8,
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo"),
    )
    gpt_3_5_turbo_proxy = LLMModel(
        name="gpt-3.5-turbo",
        api_url="https://whocars123-oai-proxy.hf.space/proxy/openai/v1/chat/completions",
        max_total_tokens=4096,
        max_tokens_per_request=4096,
        token_margin=8,
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo"),
    )

    gpt_4 = LLMModel(
        name="gpt-4",
        api_url="https://api.openai.com/v1/chat/completions",
        max_total_tokens=8192,
        max_tokens_per_request=8192,
        token_margin=8,
        tokenizer=tiktoken.encoding_for_model("gpt-4"),
    )

    gpt_4_proxy = LLMModel(
        name="gpt-4",
        api_url="https://whocars123-oai-proxy.hf.space/proxy/openai/v1/chat/completions",
        max_total_tokens=8192,
        max_tokens_per_request=8192,
        token_margin=8,
        tokenizer=tiktoken.encoding_for_model("gpt-4"),
    )


@dataclass
class MessageHistory:  # message history for user and gpt
    role: str
    content: str
    tokens: int
    is_user: bool
    timestamp: int = field(default_factory=lambda: UTC.timestamp(hour_diff=9))
    uuid: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self):
        self.role = GptRoles.get_value(self.role).lower()

    def __repr__(self) -> str:
        return f'<{self.role} uuid="{self.uuid}" date="{self.datetime}" tokens="{self.tokens}">{self.content}</>'

    @property
    def datetime(self) -> datetime:
        return UTC.timestamp_to_datetime(self.timestamp)


@dataclass
class UserGptProfile:  # user gpt profile for user and gpt
    user_id: str
    chat_room_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: int = field(default_factory=lambda: UTC.timestamp(hour_diff=9))
    user_role: str = field(default=GptRoles.USER.value)
    gpt_role: str = field(default=GptRoles.GPT.value)
    system_role: str = field(default=GptRoles.SYSTEM.value)
    temperature: float = 0.9
    top_p: float = 1.0
    presence_penalty: float = 0
    frequency_penalty: float = 0


@dataclass
class UserGptContext:  # user gpt context for user and gpt
    user_gpt_profile: UserGptProfile
    gpt_model: OpenAIModels
    user_message_histories: list[MessageHistory] = field(default_factory=list)
    gpt_message_histories: list[MessageHistory] = field(default_factory=list)
    system_message_histories: list[MessageHistory] = field(default_factory=list)
    user_message_tokens: int = field(init=False, default=0)
    gpt_message_tokens: int = field(init=False, default=0)
    system_message_tokens: int = field(init=False, default=0)

    optional_info: dict = field(default_factory=dict)

    def __post_init__(self):
        for role in tuple(key.split("_")[0] for key in self.__annotations__.keys() if "message_tokens" in key.lower()):
            setattr(self, f"{role}_message_tokens", sum([m.tokens for m in getattr(self, f"{role}_message_histories")]))

    @classmethod
    def parse_stringified_json(cls, stred_json: str) -> "UserGptContext":
        stored: dict = orjson.loads(stred_json)
        return cls(
            user_gpt_profile=UserGptProfile(**stored["user_gpt_profile"]),
            gpt_model=OpenAIModels._member_map_[stored["gpt_model"].replace(".", "_").replace("-", "_")],  # type: ignore
            user_message_histories=[MessageHistory(**m) for m in stored["user_message_histories"]],
            gpt_message_histories=[MessageHistory(**m) for m in stored["gpt_message_histories"]],
            system_message_histories=[MessageHistory(**m) for m in stored["system_message_histories"]],
        )

    def json(self) -> dict:
        return {
            "user_gpt_profile": asdict(self.user_gpt_profile),
            "gpt_model": self.gpt_model.name,
            "user_message_histories": [m.__dict__ for m in self.user_message_histories],
            "gpt_message_histories": [m.__dict__ for m in self.gpt_message_histories],
            "system_message_histories": [m.__dict__ for m in self.system_message_histories],
        }

    def to_stringified_json(self) -> str:
        return orjson.dumps(self.json()).decode("utf-8")

    def tokenize(self, message: str) -> list[int]:
        return self.gpt_model.value.tokenizer.encode(message)

    @property
    def left_tokens(self) -> int:
        return self.gpt_model.value.max_total_tokens - self.total_tokens - self.gpt_model.value.token_margin

    @property
    def total_tokens(self) -> int:
        return self.user_message_tokens + self.gpt_message_tokens + self.system_message_tokens

    @property
    def token_per_request(self) -> int:
        return self.gpt_model.value.max_tokens_per_request

    @property
    def user_id(self) -> str:
        return self.user_gpt_profile.user_id

    @property
    def chat_room_id(self) -> str:
        return self.user_gpt_profile.chat_room_id

    def __repr__(self) -> str:
        gpt_model: OpenAIModels = self.gpt_model
        time_string: str = datetime.strptime(
            str(self.user_gpt_profile.created_at),
            "%Y%m%d%H%M%S",
        ).strftime("%Y-%m-%d %H:%M:%S")
        return f"""# User Info
- Your ID: `{self.user_id}`
- This chatroom ID: `{self.chat_room_id}`
- Your profile created at: `{time_string}`

# LLM Info
- Model Name: `{gpt_model.name}`
- Actual Model Name: `{gpt_model.value.name}`
- Temperature: `{self.user_gpt_profile.temperature}`
- Top P: `{self.user_gpt_profile.top_p}`
- Presence Penalty: `{self.user_gpt_profile.presence_penalty}`
- Frequency Penalty: `{self.user_gpt_profile.frequency_penalty}`

# Token Info
- Maximum Token Limit: `{gpt_model.value.max_total_tokens}`
- User Token Consumed: `{self.user_message_tokens}`
- GPT Token Consumed: `{self.gpt_message_tokens}`
- System Token Consumed: `{self.system_message_tokens}`
- Total Token Consumed: `{self.total_tokens}`
- Remaining Tokens: `{self.left_tokens}`

# Message Histories
- User Message_Histories={self.user_message_histories}

- GPT Message Histories={self.gpt_message_histories}

- System Message Histories={self.system_message_histories}
"""

    @classmethod
    def construct_default(
        cls,
        user_id: str,
        chat_room_id: str,
        gpt_model: OpenAIModels = OpenAIModels.gpt_3_5_turbo,
    ):
        return cls(
            user_gpt_profile=UserGptProfile(user_id=user_id, chat_room_id=chat_room_id),
            gpt_model=gpt_model,
        )

    def reset(self):
        for k, v in self.construct_default(
            self.user_id,
            self.chat_room_id,
            self.gpt_model,
        ).__dict__.items():
            setattr(self, k, v)

    def ensure_token_not_exceed(
        self,
    ) -> int:
        deleted_histories: int = 0
        while self.left_tokens <= 0:
            deleted_histories += 1
            emptied_user_tokens: int = self.user_message_histories.pop(0).tokens
            emptied_gpt_tokens: int = self.gpt_message_histories.pop(0).tokens
            self.user_message_tokens -= emptied_user_tokens
            self.gpt_message_tokens -= emptied_gpt_tokens
        return deleted_histories
