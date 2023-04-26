from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from uuid import uuid4
from typing import Type

import orjson
import tiktoken

from app.utils.date_utils import UTC


class GptRoles(Enum):
    GPT = "assistant"
    SYSTEM = "system"
    USER = "user"

    @classmethod
    def get_name(cls, role: Type["GptRoles"] | str) -> str:
        if isinstance(role, cls):  # when role is member
            return role.name
        elif role in cls._value2member_map_:  # when role is value
            return cls._value2member_map_[role].name
        elif role.upper() in cls._member_map_:  # when role is name
            return role
        else:
            raise ValueError(f"Invalid role: {role}")

    @classmethod
    def get_value(cls, role: Type["GptRoles"] | str) -> str:
        if isinstance(role, cls):  # when role is member
            return role.value
        elif role in cls._value2member_map_:  # when role is value
            return role
        elif role.upper() in cls._member_map_:  # when role is name
            return cls._member_map_[role.upper()].value
        else:
            raise ValueError(f"Invalid role: {role}")

    @classmethod
    def get_member(cls, role: Type["GptRoles"] | str) -> Type["GptRoles"]:
        if isinstance(role, cls):  # when role is member
            return role
        elif role in cls._value2member_map_:  # when role is value
            return cls._value2member_map_[role]
        elif role.upper() in cls._member_map_:  # when role is name
            return cls._member_map_[role.upper()]
        else:
            raise ValueError(f"Invalid role: {role}")


@dataclass
class GptModel:  # gpt model for openai api
    name: str  # model name for openai api
    max_total_tokens: int  # max total tokens for openai api
    max_tokens_per_request: int  # max tokens per request for openai api
    token_margin: int = 8  # token margin for openai api
    api_url: str = "https://api.openai.com/v1/chat/completions"  # api url for openai
    tokenizer: tiktoken.core.Encoding = field(init=False)

    def __post_init__(self):
        self.tokenizer = tiktoken.encoding_for_model(self.name)


@dataclass
class GptModels:  # gpt models for openai api
    gpt_3_5_turbo: GptModel = field(
        default_factory=lambda: GptModel(
            name="gpt-3.5-turbo",
            api_url="https://api.openai.com/v1/chat/completions",
            max_total_tokens=4096,
            max_tokens_per_request=4096,
            token_margin=8,
        )
    )
    gpt_4: GptModel = field(
        default_factory=lambda: GptModel(
            name="gpt-4",
            api_url="https://api.openai.com/v1/chat/completions",
            max_total_tokens=8192,
            max_tokens_per_request=8192,
            token_margin=8,
        )
    )


@dataclass
class MessageHistory:  # message history for user and gpt
    role: str
    content: str
    tokens: int
    is_user: bool
    timestamp: int = field(default_factory=lambda: UTC.timestamp(hour_diff=9))
    uuid: str = field(default_factory=lambda: str(uuid4()))

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
    gpt_model: GptModel
    user_message_histories: list[MessageHistory] = field(default_factory=list)
    gpt_message_histories: list[MessageHistory] = field(default_factory=list)
    system_message_histories: list[MessageHistory] = field(default_factory=list)
    user_message_tokens: int = field(init=False, default=0)
    gpt_message_tokens: int = field(init=False, default=0)
    system_message_tokens: int = field(init=False, default=0)
    is_discontinued: bool = False

    def __post_init__(self):
        for role in tuple(key.split("_")[0] for key in self.__annotations__.keys() if "message_tokens" in key.lower()):
            setattr(self, f"{role}_message_tokens", sum([m.tokens for m in getattr(self, f"{role}_message_histories")]))

    @classmethod
    def parse_stringified_json(cls, stred_json: str) -> "UserGptContext":
        stored: dict = orjson.loads(stred_json)
        return cls(
            user_gpt_profile=UserGptProfile(**stored["user_gpt_profile"]),
            gpt_model=getattr(GPT_MODELS, stored["gpt_model"].replace(".", "_").replace("-", "_")),
            user_message_histories=[MessageHistory(**m) for m in stored["user_message_histories"]],
            gpt_message_histories=[MessageHistory(**m) for m in stored["gpt_message_histories"]],
            system_message_histories=[MessageHistory(**m) for m in stored["system_message_histories"]],
            is_discontinued=stored["is_discontinued"],
        )

    def to_json(self) -> dict:
        return {
            "user_gpt_profile": asdict(self.user_gpt_profile),
            "gpt_model": self.gpt_model.name,
            "user_message_histories": [m.__dict__ for m in self.user_message_histories],
            "gpt_message_histories": [m.__dict__ for m in self.gpt_message_histories],
            "system_message_histories": [m.__dict__ for m in self.system_message_histories],
            "is_discontinued": self.is_discontinued,
        }

    def to_stringified_json(self) -> str:
        return orjson.dumps(self.json())

    def tokenize(self, message: str) -> list[int]:
        return self.gpt_model.tokenizer.encode(message)

    @property
    def left_tokens(self) -> int:
        return self.gpt_model.max_total_tokens - self.total_tokens - self.gpt_model.token_margin

    @property
    def total_tokens(self) -> int:
        return self.user_message_tokens + self.gpt_message_tokens + self.system_message_tokens

    @property
    def token_per_request(self) -> int:
        return self.gpt_model.max_tokens_per_request

    def __repr__(self) -> str:
        m: GptModel = self.gpt_model
        return f"""[User Info]
user_id={self.user_gpt_profile.user_id}

[GPT Info]
gpt_model={m.name}
temperature={self.user_gpt_profile.temperature}
top_p={self.user_gpt_profile.top_p}
presence_penalty={self.user_gpt_profile.presence_penalty}
frequency_penalty={self.user_gpt_profile.frequency_penalty}

[Message Histories]
user_message_histories={self.user_message_histories}

gpt_message_histories={self.gpt_message_histories}

system_message_histories={self.system_message_histories}

[Token Info]
Token limit: {m.max_total_tokens}
User token consumed={self.user_message_tokens}
Gpt token consumed={self.gpt_message_tokens}
System token consumed={self.system_message_tokens}
Total token consumed: {self.total_tokens}
Remaining token: {self.left_tokens}"""

    @classmethod
    def construct_default(cls, user_id: str, gpt_model_name: str = "gpt-3.5-turbo"):
        return cls(
            user_gpt_profile=UserGptProfile(user_id=user_id),
            gpt_model=getattr(GPT_MODELS, gpt_model_name.replace(".", "_").replace("-", "_")),
        )

    def reset(self):
        for k, v in self.construct_default(self.user_gpt_profile.user_id, self.gpt_model.name).__dict__.items():
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


GPT_MODELS = GptModels()
