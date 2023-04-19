from sqlalchemy import (
    String,
    Enum,
    ForeignKey,
    Float,
    Text,
)
from sqlalchemy.orm import (
    relationship,
    Mapped,
    mapped_column,
)
from app.database.schemas.auth import Users
from .. import Base
from . import Mixin


class ChatRooms(Base, Mixin):
    __tablename__ = "chat_rooms"
    uuid: Mapped[str] = mapped_column(String(length=36), index=True, unique=True)
    status: Mapped[str] = mapped_column(Enum("active", "deleted", "blocked"), default="active")
    chat_room_type: Mapped[str] = mapped_column(String(length=20), index=True)
    name: Mapped[str] = mapped_column(String(length=20))
    description: Mapped[str | None] = mapped_column(String(length=100))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    users: Mapped["Users"] = relationship(back_populates="chat_rooms")
    chat_messages: Mapped["ChatMessages"] = relationship(back_populates="chat_rooms", cascade="all, delete-orphan")


class ChatMessages(Base, Mixin):
    __tablename__ = "chat_messages"
    uuid: Mapped[str] = mapped_column(String(length=36), index=True, unique=True)
    status: Mapped[str] = mapped_column(Enum("active", "deleted", "blocked"), default="active")
    role: Mapped[str] = mapped_column(String(length=20), default="user")
    message: Mapped[str] = mapped_column(Text)
    chat_room_id: Mapped[int] = mapped_column(ForeignKey("chat_rooms.id"))
    chat_rooms: Mapped["ChatRooms"] = relationship(back_populates="chat_messages")
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    users: Mapped["Users"] = relationship(back_populates="chat_messages")


class GptPresets(Base, Mixin):
    __tablename__ = "gpt_presets"
    temperature: Mapped[float] = mapped_column(Float, default=0.9)
    top_p: Mapped[float] = mapped_column(Float, default=1.0)
    presence_penalty: Mapped[float] = mapped_column(Float, default=0)
    frequency_penalty: Mapped[float] = mapped_column(Float, default=0)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    users: Mapped["Users"] = relationship(back_populates="gpt_presets")
