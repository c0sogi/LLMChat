import enum
from sqlalchemy import (
    String,
    Integer,
    Enum,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import (
    relationship,
    Mapped,
    mapped_column,
)

from app.viewmodels.status import ApiKeyStatus, UserStatus
from .. import Base
from . import TableMixin


class Users(Base, TableMixin):
    __tablename__ = "users"
    status: Mapped[str] = mapped_column(Enum(UserStatus), default=UserStatus.active)
    email: Mapped[str] = mapped_column(String(length=50))
    password: Mapped[str | None] = mapped_column(String(length=100))
    name: Mapped[str | None] = mapped_column(String(length=20))
    phone_number: Mapped[str | None] = mapped_column(String(length=20))
    profile_img: Mapped[str | None] = mapped_column(String(length=100))
    marketing_agree: Mapped[bool] = mapped_column(Boolean, default=True)
    api_keys: Mapped["ApiKeys"] = relationship(
        back_populates="users", cascade="all, delete-orphan", lazy=True
    )
    # chat_rooms: Mapped["ChatRooms"] = relationship(back_populates="users", cascade="all, delete-orphan", lazy=True)
    # chat_messages: Mapped["ChatMessages"] = relationship(
    #     back_populates="users", cascade="all, delete-orphan", lazy=True
    # )
    # gpt_presets: Mapped["GptPresets"] = relationship(
    #     back_populates="users", cascade="all, delete-orphan", lazy=True, uselist=False
    # )


class ApiKeys(Base, TableMixin):
    __tablename__ = "api_keys"
    status: Mapped[str] = mapped_column(Enum(ApiKeyStatus), default=ApiKeyStatus.active)
    access_key: Mapped[str] = mapped_column(String(length=64), index=True, unique=True)
    secret_key: Mapped[str] = mapped_column(String(length=64))
    user_memo: Mapped[str | None] = mapped_column(String(length=40))
    is_whitelisted: Mapped[bool] = mapped_column(default=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    users: Mapped["Users"] = relationship(back_populates="api_keys")
    whitelists: Mapped["ApiWhiteLists"] = relationship(
        backref="api_keys", cascade="all, delete-orphan"
    )


class ApiWhiteLists(Base, TableMixin):
    __tablename__ = "api_whitelists"
    api_key_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("api_keys.id", ondelete="CASCADE")
    )
    ip_address: Mapped[str] = mapped_column(String(length=64))


# class ChatRooms(Base, Mixin):
#     __tablename__ = "chat_rooms"
#     uuid: Mapped[str] = mapped_column(String(length=36), index=True, unique=True)
#     status: Mapped[str] = mapped_column(Enum("active", "deleted", "blocked"), default="active")
#     chat_room_type: Mapped[str] = mapped_column(String(length=20), index=True)
#     name: Mapped[str] = mapped_column(String(length=20))
#     description: Mapped[str | None] = mapped_column(String(length=100))
#     user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
#     users: Mapped["Users"] = relationship(back_populates="chat_rooms")
#     chat_messages: Mapped["ChatMessages"] = relationship(back_populates="chat_rooms", cascade="all, delete-orphan")


# class ChatMessages(Base, Mixin):
#     __tablename__ = "chat_messages"
#     uuid: Mapped[str] = mapped_column(String(length=36), index=True, unique=True)
#     status: Mapped[str] = mapped_column(Enum("active", "deleted", "blocked"), default="active")
#     role: Mapped[str] = mapped_column(String(length=20), default="user")
#     message: Mapped[str] = mapped_column(Text)
#     chat_room_id: Mapped[int] = mapped_column(ForeignKey("chat_rooms.id", ondelete="CASCADE"))
#     chat_rooms: Mapped["ChatRooms"] = relationship(back_populates="chat_messages")
#     user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
#     users: Mapped["Users"] = relationship(back_populates="chat_messages")


# class GptPresets(Base, Mixin):
#     __tablename__ = "gpt_presets"
#     temperature: Mapped[float] = mapped_column(Float, default=0.9)
#     top_p: Mapped[float] = mapped_column(Float, default=1.0)
#     presence_penalty: Mapped[float] = mapped_column(Float, default=0)
#     frequency_penalty: Mapped[float] = mapped_column(Float, default=0)
#     user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
#     users: Mapped["Users"] = relationship(back_populates="gpt_presets")
