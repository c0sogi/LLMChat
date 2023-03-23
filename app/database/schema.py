from typing import Optional
from sqlalchemy import (
    Column,
    func,
    String,
    Integer,
    Enum,
    Boolean,
    ForeignKey,
    Select,
    select,
    update,
    LargeBinary,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import (
    relationship,
    Mapped,
    mapped_column,
)
from datetime import datetime
from app.database.connection import Base, SQLAlchemy
from app.common.config import config

db = SQLAlchemy()
# ========================== Schema section begins ==========================


class Mixin:
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(default=func.utc_timestamp())
    updated_at: Mapped[datetime] = mapped_column(
        default=func.utc_timestamp(),
        onupdate=func.utc_timestamp(),
    )
    ip_address: Mapped[str | None] = mapped_column(String(length=40))

    def set_values_as(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def all_columns(self) -> list[Column]:
        return [
            col
            for col in self.__table__.columns
            if (not col.primary_key) and (col.name != "created_at")
        ]

    @classmethod
    async def add_all(
        cls,
        *args: dict,
        autocommit: bool = False,
        refresh: bool = False,
        session: AsyncSession | None = None,
    ) -> list[Base]:
        return await db.add_all(
            cls, *args, autocommit=autocommit, refresh=refresh, session=session
        )

    @classmethod
    async def add_one(
        cls,
        autocommit: bool = False,
        refresh: bool = False,
        session: AsyncSession | None = None,
        **kwargs: any,
    ) -> Base:
        return await db.add(
            cls, autocommit=autocommit, refresh=refresh, session=session, **kwargs
        )

    @classmethod
    async def update_where(
        cls,
        filter_by: dict,
        updated: dict,
        autocommit: bool = False,
        refresh: bool = False,
        session: AsyncSession | None = None,
    ) -> Base:
        stmt = update(cls).filter_by(**filter_by).values(**updated)
        return await db.run_in_session(db._execute)(
            session, autocommit=autocommit, refresh=refresh, stmt=stmt
        )

    @classmethod
    async def fetchall_filtered_by(
        cls, session: AsyncSession | None = None, **kwargs: any
    ) -> list[Base]:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__fetchall(stmt=stmt, session=session)

    @classmethod
    async def one_filtered_by(
        cls, session: AsyncSession | None = None, **kwargs: any
    ) -> Base:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__one(stmt=stmt, session=session)

    @classmethod
    async def first_filtered_by(
        cls, session: AsyncSession | None = None, **kwargs: any
    ) -> Base:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__first(stmt=stmt, session=session)

    @classmethod
    async def one_or_none_filtered_by(
        cls, session: AsyncSession | None = None, **kwargs: any
    ) -> Optional[Base]:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__one_or_none(stmt=stmt, session=session)

    @classmethod
    async def fetchall_filtered(
        cls, *criteria: bool, session: AsyncSession | None = None
    ) -> list[Base]:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__fetchall(stmt=stmt, session=session)

    @classmethod
    async def one_filtered(
        cls, *criteria: bool, session: AsyncSession | None = None
    ) -> Base:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__one(stmt=stmt, session=session)

    @classmethod
    async def first_filtered(
        cls, *criteria: bool, session: AsyncSession | None = None
    ) -> Base:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__first(stmt=stmt, session=session)

    @classmethod
    async def one_or_none_filtered(
        cls, *criteria: bool, session: AsyncSession | None = None
    ) -> Optional[Base]:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__one_or_none(stmt=stmt, session=session)


class Users(Base, Mixin):
    __tablename__ = "users"
    status: Mapped[str] = mapped_column(
        Enum("active", "deleted", "blocked"), default="active"
    )
    email: Mapped[str] = mapped_column(String(length=20))
    password: Mapped[str | None] = mapped_column(String(length=72))
    name: Mapped[str | None] = mapped_column(String(length=20))
    phone_number: Mapped[str | None] = mapped_column(String(length=20))
    profile_img: Mapped[str | None] = mapped_column(String(length=100))
    marketing_agree: Mapped[bool] = mapped_column(Boolean, default=True)
    api_keys = relationship(
        "ApiKeys", back_populates="users", cascade="all, delete-orphan"
    )


class ApiKeys(Base, Mixin):
    __tablename__ = "api_keys"
    status: Mapped[str] = mapped_column(
        Enum("active", "stopped", "deleted"), default="active"
    )
    access_key: Mapped[str] = mapped_column(String(length=64), index=True, unique=True)
    secret_key: Mapped[str] = mapped_column(String(length=64))
    user_memo: Mapped[str | None] = mapped_column(String(length=40))
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


class Debugging(Base):
    __tablename__ = "debugging"
    id: Mapped[int] = mapped_column(primary_key=True)
    data: Mapped[bytes] = mapped_column(LargeBinary)


class ClientLog(Base):
    __tablename__ = "client_log"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(default=func.utc_timestamp())
    visited_url: Mapped[str] = mapped_column(String(length=100))


# ========================== Schema section ends ==========================
db.init(config=config)


# json_data = {
#     "url": request.url.hostname + request.url.path,
#     "method": str(request.method),
#     "statusCode": status_code,
#     "errorDetail": await error_log_generator(error=error, request=request)
#     if error is not None
#     else None,
#     "client": {
#         "client": request.state.ip,
#         "user": user.id if user and user.id else None,
#         "email": (await hide_email(email=user.email))
#         if user and user.email
#         else None,
#     },
#     "processedTime": str(round(processed_time * 1000, 5)) + "ms",
#     "datetimeUTC": utc_now.strftime("%Y/%m/%d %H:%M:%S"),
#     "datetimeKST": (utc_now + timedelta(hours=9)).strftime("%Y/%m/%d %H:%M:%S"),
# } | kwargs
