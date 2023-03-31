from datetime import datetime
from enum import Enum

from pydantic import Field
from pydantic.main import BaseModel

# from pydantic.networks import EmailStr, IPvanyAddress


class UserRegister(BaseModel):
    email: str = None
    password: str = None


class SnsType(str, Enum):
    email: str = "email"
    facebook: str = "facebook"
    google: str = "google"
    kakao: str = "kakao"


class Token(BaseModel):
    Authorization: str = None


class EmailRecipients(BaseModel):
    name: str
    email: str


class SendEmail(BaseModel):
    email_to: list[EmailRecipients] = None


class KakaoMsgBody(BaseModel):
    msg: str = None


class MessageOk(BaseModel):
    message: str = Field(default="OK")


class UserToken(BaseModel):
    id: int
    email: str = None
    name: str = None
    phone_number: str = None
    profile_img: str = None
    sns_type: str = None

    class Config:
        orm_mode = True


class UserMe(BaseModel):
    id: int
    email: str = None
    name: str = None
    phone_number: str = None
    profile_img: str = None
    sns_type: str = None

    class Config:
        orm_mode = True


class AddApiKey(BaseModel):
    user_memo: str = None

    class Config:
        orm_mode = True


class GetApiKeyList(AddApiKey):
    id: int = None
    access_key: str = None
    created_at: datetime = None


class GetApiKeys(GetApiKeyList):
    secret_key: str = None


class CreateApiWhiteLists(BaseModel):
    ip_address: str = None


class GetApiWhiteLists(CreateApiWhiteLists):
    id: int

    class Config:
        orm_mode = True


class CreateChatMessage(BaseModel):
    message: str
    role: str
    user_id: int
    chat_room_id: int

    class Config:
        orm_mode = True


class SendChatMessage(BaseModel):
    msg: str
    finish: bool
    chat_room_id: int
    is_user: bool

    class Config:
        orm_mode = True


class ReceiveChatMessage(BaseModel):
    msg: str
    translate: bool
    chat_room_id: int

    class Config:
        orm_mode = True


class CreateChatRoom(BaseModel):
    chat_room_type: str
    name: str
    description: str = None
    user_id: int

    class Config:
        orm_mode = True
