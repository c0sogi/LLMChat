from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union, Type
from uuid import uuid4

from pydantic import Field
from pydantic.main import BaseModel

from app.database.schemas.auth import UserStatus
from app.utils.date_utils import UTC

JSON_TYPES = Union[int, float, str, bool, dict, list, None]


class UserRegister(BaseModel):
    email: str
    password: str


class SnsType(str, Enum):
    EMAIL = "email"
    FACEBOOK = "facebook"
    GOOGLE = "google"
    KAKAO = "kakao"


class Token(BaseModel):
    Authorization: str


class EmailRecipients(BaseModel):
    name: str
    email: str


class SendEmail(BaseModel):
    email_to: list[EmailRecipients]


class KakaoMsgBody(BaseModel):
    msg: str


class MessageOk(BaseModel):
    message: str = Field(default="OK")


class UserToken(BaseModel):
    id: int
    status: UserStatus
    email: Optional[str] = None
    name: Optional[str] = None

    class Config:
        orm_mode = True


class UserMe(BaseModel):
    id: int
    email: Optional[str] = None
    name: Optional[str] = None
    phone_number: Optional[str] = None
    profile_img: Optional[str] = None
    sns_type: Optional[str] = None

    class Config:
        orm_mode = True


class AddApiKey(BaseModel):
    user_memo: Optional[str] = None

    class Config:
        orm_mode = True


class GetApiKey(AddApiKey):
    id: int
    access_key: str
    created_at: datetime


class GetApiKeyFirstTime(GetApiKey):
    secret_key: str


class CreateApiWhiteList(BaseModel):
    ip_address: str


class GetApiWhiteList(CreateApiWhiteList):
    id: int

    class Config:
        orm_mode = True


class CreateChatMessage(BaseModel):  # stub
    message: str
    role: str
    user_id: int
    chat_room_id: str

    class Config:
        orm_mode = True


class MessageToWebsocket(BaseModel):
    msg: Optional[str]
    finish: bool
    chat_room_id: Optional[str] = None
    actual_role: Optional[str] = None
    init: bool = False
    model_name: Optional[str] = None
    uuid: Optional[str] = None
    wait_next_query: Optional[bool] = None

    class Config:
        orm_mode = True


class MessageFromWebsocket(BaseModel):
    msg: str
    translate: Optional[str] = None
    chat_room_id: str


class CreateChatRoom(BaseModel):  # stub
    chat_room_type: str
    name: str
    description: Optional[str] = None
    user_id: int

    class Config:
        orm_mode = True


class OpenAIChatMessage(BaseModel):
    role: str
    content: str

    class Config:
        orm_mode = True


class MessageHistory(BaseModel):
    role: str
    content: str
    tokens: int
    actual_role: str
    timestamp: int = Field(default_factory=UTC.timestamp)
    uuid: str = Field(default_factory=lambda: uuid4().hex)
    model_name: Optional[str] = None
    summarized: Optional[str] = None
    summarized_tokens: Optional[int] = None

    class Config:
        orm_mode = True

    def __repr__(self) -> str:
        if self.summarized is not None:
            return (
                f'<{self.role} uuid="{self.uuid}" date="{self.datetime}Z" tokens="{self.tokens}" '
                f'summarized="{self.summarized}">{self.content}</>'
            )
        return f'<{self.role} uuid="{self.uuid}" date="{self.datetime}Z" tokens="{self.tokens}">{self.content}</>'

    @property
    def datetime(self) -> datetime:
        return UTC.timestamp_to_datetime(self.timestamp)


class InitMessage(BaseModel):
    previous_chats: Optional[list[dict]] = None
    chat_rooms: Optional[list[dict[str, str]]] = None
    models: Optional[list[str]] = None
    selected_model: Optional[str] = None
    tokens: Optional[int] = None


class StreamProgress(BaseModel):
    response: str = ""
    buffer: str = ""
    uuid: Optional[str] = None


class UserChatRoles(BaseModel):
    ai: str
    system: str
    user: str


class SummarizedResult(BaseModel):
    user_id: str
    chat_room_id: str
    role: str
    content: str
    uuid: str


def get_json_type(python_type: Type[JSON_TYPES]) -> str:
    """Returns the JSON type for a given python type"""
    if python_type is int:
        return "integer"
    elif python_type is float:
        return "number"
    elif python_type is str:
        return "string"
    elif python_type is bool:
        return "boolean"
    elif python_type is dict:
        return "object"
    elif python_type is list:
        return "array"
    else:
        return "null"


class OpenAIFunctionParameter(BaseModel):
    name: str
    type: Type[JSON_TYPES]
    description: Optional[str] = None
    enum: Optional[list[JSON_TYPES]] = None

    def to_dict(self):
        param_dict: dict[str, Any] = {"type": get_json_type(self.type)}
        if self.description:
            param_dict["description"] = self.description
        if self.enum:
            param_dict["enum"] = self.enum
        return {self.name: param_dict}


class OpenAIFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[list[OpenAIFunctionParameter]] = None
    required: Optional[list[str]] = None

    def to_dict(self):
        function_dict: dict[str, Any] = {"name": self.name}
        if self.description:
            function_dict["description"] = self.description
        if self.parameters:
            function_dict["parameters"] = {
                "type": "object",
                "properties": {
                    param.name: param.to_dict()[param.name] for param in self.parameters
                },
                "required": [
                    param.name
                    for param in self.parameters
                    if param.name in (self.required or [])
                ],
            }
        return function_dict
