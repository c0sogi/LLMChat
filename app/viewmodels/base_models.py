from datetime import datetime
from enum import Enum

from pydantic import Field
from pydantic.main import BaseModel

# from pydantic.networks import EmailStr, IPvanyAddress


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
    email: str | None = None
    name: str | None = None
    phone_number: str | None = None
    profile_img: str | None = None
    sns_type: str | None = None

    class Config:
        orm_mode = True


class UserMe(BaseModel):
    id: int
    email: str | None = None
    name: str | None = None
    phone_number: str | None = None
    profile_img: str | None = None
    sns_type: str | None = None

    class Config:
        orm_mode = True


class AddApiKey(BaseModel):
    user_memo: str | None = None

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
    msg: str | None
    finish: bool
    chat_room_id: str
    is_user: bool
    init: bool = False
    model_name: str | None = None

    class Config:
        orm_mode = True


class MessageFromWebsocket(BaseModel):
    msg: str
    translate: bool
    chat_room_id: str


class CreateChatRoom(BaseModel):  # stub
    chat_room_type: str
    name: str
    description: str | None = None
    user_id: int

    class Config:
        orm_mode = True


class SendToStream(BaseModel):
    role: str
    content: str

    class Config:
        orm_mode = True


class SendInitToWebsocket(BaseModel):
    content: str
    tokens: int
    is_user: bool
    timestamp: int
    model_name: str | None = None

    class Config:
        orm_mode = True


class InitMessage(BaseModel):
    previous_chats: list[dict]
    chat_room_ids: list[str]
