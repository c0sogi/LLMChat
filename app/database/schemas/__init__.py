from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, Select, String, func, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql._typing import _ColumnExpressionArgument
from sqlalchemy.sql.roles import ExpressionElementRole

from .. import DeclarativeMeta
from ..connection import db


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

    def to_dict(self, *args, exclude: list | None = None):
        q_dict: dict = {}
        for c in self.__table__.columns:  # type: ignore
            if not args or c.name in args:
                if not exclude or c.name not in exclude:
                    q_dict[c.name] = getattr(self, c.name)
        return q_dict

    @property
    def columns(self) -> list[Column]:
        return [col for col in self.__table__.columns if (not col.primary_key) and (col.name != "created_at")]  # type: ignore

    @classmethod
    async def add_all(
        cls,
        *args: dict,
        autocommit: bool = False,
        refresh: bool = False,
        session: AsyncSession | None = None,
    ) -> list[DeclarativeMeta]:
        return await db.add_all(
            cls,  # type: ignore
            *args,
            autocommit=autocommit,
            refresh=refresh,
            session=session,
        )

    @classmethod
    async def add_one(
        cls,
        autocommit: bool = False,
        refresh: bool = False,
        session: AsyncSession | None = None,
        **kwargs: Any,
    ) -> DeclarativeMeta:
        return await db.add(
            cls,  # type: ignore
            autocommit=autocommit,
            refresh=refresh,
            session=session,
            **kwargs,
        )

    @classmethod
    async def update_filtered(
        cls,
        *criteria: ExpressionElementRole[bool],
        updated: dict,
        autocommit: bool = False,
        refresh: bool = False,
        session: AsyncSession | None = None,
    ) -> DeclarativeMeta:
        stmt = update(cls).filter(*criteria).values(**updated)
        return await db.run_in_session(db._execute)(
            session,
            autocommit=autocommit,
            refresh=refresh,
            stmt=stmt,
        )

    @classmethod
    async def delete_filtered(
        cls,
        *criteria: ExpressionElementRole[bool],
        autocommit: bool = False,
        refresh: bool = False,
        session: AsyncSession | None = None,
    ) -> DeclarativeMeta:
        stmt = delete(cls).filter(*criteria)
        return await db.run_in_session(db._execute)(
            session,
            autocommit=autocommit,
            refresh=refresh,
            stmt=stmt,
        )

    @classmethod
    async def fetchall_filtered_by(cls, session: AsyncSession | None = None, **kwargs: Any) -> list[DeclarativeMeta]:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__fetchall(stmt=stmt, session=session)

    @classmethod
    async def one_filtered_by(cls, session: AsyncSession | None = None, **kwargs: Any) -> DeclarativeMeta:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__one(stmt=stmt, session=session)

    @classmethod
    async def first_filtered_by(cls, session: AsyncSession | None = None, **kwargs: Any) -> DeclarativeMeta:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__first(stmt=stmt, session=session)

    @classmethod
    async def one_or_none_filtered_by(
        cls, session: AsyncSession | None = None, **kwargs: Any
    ) -> Optional[DeclarativeMeta]:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__one_or_none(stmt=stmt, session=session)

    @classmethod
    async def fetchall_filtered(
        cls, *criteria: _ColumnExpressionArgument[bool], session: AsyncSession | None = None
    ) -> list[DeclarativeMeta]:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__fetchall(stmt=stmt, session=session)

    @classmethod
    async def one_filtered(
        cls, *criteria: _ColumnExpressionArgument[bool], session: AsyncSession | None = None
    ) -> DeclarativeMeta:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__one(stmt=stmt, session=session)

    @classmethod
    async def first_filtered(
        cls, *criteria: _ColumnExpressionArgument[bool], session: AsyncSession | None = None
    ) -> DeclarativeMeta:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__first(stmt=stmt, session=session)

    @classmethod
    async def one_or_none_filtered(
        cls, *criteria: _ColumnExpressionArgument[bool], session: AsyncSession | None = None
    ) -> Optional[DeclarativeMeta]:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__one_or_none(stmt=stmt, session=session)
