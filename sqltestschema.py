from typing import Optional, List
from sqlalchemy import (
    text,
    Column,
    func,
    String,
    Integer,
    select,
    Enum,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    Mapped,
    mapped_column,
)
from datetime import datetime


Base = declarative_base()


class Mixin:
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(default=func.utc_timestamp())
    updated_at: Mapped[datetime] = mapped_column(
        default=func.utc_timestamp(),
        onupdate=func.utc_timestamp(),
    )
    ip_address: Mapped[Optional[str]] = mapped_column(String(length=40))

    @property
    def all_columns(self) -> List[Column]:
        return [
            col
            for col in self.__table__.columns
            if (not col.primary_key) and (col.name != "created_at")
        ]


class Users(Base, Mixin):
    __tablename__ = "users"
    status: Mapped[str] = mapped_column(
        Enum("active", "deleted", "blocked"), default="active"
    )
    username: Mapped[str] = mapped_column(String(length=20), unique=True)
    password: Mapped[Optional[str]] = mapped_column(String(length=72))
    name: Mapped[Optional[str]] = mapped_column(String(length=20))
    phone_number: Mapped[Optional[str]] = mapped_column(String(length=20))
    profile_img: Mapped[Optional[str]] = mapped_column(String(length=100))
    marketing_agree: Mapped[bool] = mapped_column(Boolean, default=True)
    api_keys = relationship(
        "ApiKeys", back_populates="users", cascade="all, delete-orphan"
    )


class ApiKeys(Base, Mixin):
    __tablename__ = "api_keys"
    status: Mapped[str] = mapped_column(
        Enum("active", "stopped", "deleted"), default="active"
    )
    access_key: Mapped[str] = mapped_column(String(length=64), index=True)
    secret_key: Mapped[str] = mapped_column(String(length=64))
    user_memo: Mapped[Optional[str]] = mapped_column(String(length=40))
    is_whitelisted: Mapped[bool] = mapped_column(default=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    users: Mapped["Users"] = relationship(back_populates="api_keys")
    whitelists: Mapped["ApiWhiteLists"] = relationship(
        backref="api_keys", cascade="all, delete-orphan"
    )


class ApiWhiteLists(Base, Mixin):
    __tablename__ = "api_whitelists"
    api_key_id: Mapped[int] = Column(Integer, ForeignKey("api_keys.id"))
    ip_address: Mapped[str] = Column(String(length=64))
