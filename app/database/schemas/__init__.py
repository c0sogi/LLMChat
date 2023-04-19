from typing import Optional
from sqlalchemy import (
    Column,
    func,
    String,
    Select,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)
from datetime import datetime
from .. import Base
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

    def to_dict(self, *args, exclude: list = None):
        q_dict: dict = {}
        for c in self.__table__.columns:
            if not args or c.name in args:
                if not exclude or c.name not in exclude:
                    q_dict[c.name] = getattr(self, c.name)
        return q_dict

    @property
    def columns(self) -> list[Column]:
        return [col for col in self.__table__.columns if (not col.primary_key) and (col.name != "created_at")]

    @classmethod
    async def add_all(
        cls,
        *args: dict,
        autocommit: bool = False,
        refresh: bool = False,
        session: AsyncSession | None = None,
    ) -> list[Base]:
        return await db.add_all(cls, *args, autocommit=autocommit, refresh=refresh, session=session)

    @classmethod
    async def add_one(
        cls,
        autocommit: bool = False,
        refresh: bool = False,
        session: AsyncSession | None = None,
        **kwargs: any,
    ) -> Base:
        return await db.add(cls, autocommit=autocommit, refresh=refresh, session=session, **kwargs)

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
        return await db.run_in_session(db._execute)(session, autocommit=autocommit, refresh=refresh, stmt=stmt)

    @classmethod
    async def fetchall_filtered_by(cls, session: AsyncSession | None = None, **kwargs: any) -> list[Base]:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__fetchall(stmt=stmt, session=session)

    @classmethod
    async def one_filtered_by(cls, session: AsyncSession | None = None, **kwargs: any) -> Base:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__one(stmt=stmt, session=session)

    @classmethod
    async def first_filtered_by(cls, session: AsyncSession | None = None, **kwargs: any) -> Base:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__first(stmt=stmt, session=session)

    @classmethod
    async def one_or_none_filtered_by(cls, session: AsyncSession | None = None, **kwargs: any) -> Optional[Base]:
        stmt: Select[tuple] = select(cls).filter_by(**kwargs)
        return await db.scalars__one_or_none(stmt=stmt, session=session)

    @classmethod
    async def fetchall_filtered(cls, *criteria: bool, session: AsyncSession | None = None) -> list[Base]:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__fetchall(stmt=stmt, session=session)

    @classmethod
    async def one_filtered(cls, *criteria: bool, session: AsyncSession | None = None) -> Base:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__one(stmt=stmt, session=session)

    @classmethod
    async def first_filtered(cls, *criteria: bool, session: AsyncSession | None = None) -> Base:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__first(stmt=stmt, session=session)

    @classmethod
    async def one_or_none_filtered(cls, *criteria: bool, session: AsyncSession | None = None) -> Optional[Base]:
        stmt: Select[tuple] = select(cls).filter(*criteria)
        return await db.scalars__one_or_none(stmt=stmt, session=session)
