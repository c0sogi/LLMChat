from typing import Optional, List
from sqlalchemy import (
    Column,
    func,
    String,
    Integer,
    Enum,
    Boolean,
    ForeignKey,
    select,
    Select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, Tuple
from uuid import uuid4
from sqlalchemy.orm import (
    relationship,
    Mapped,
    mapped_column,
)
from datetime import datetime
from sqltest import Base, SQLAlchemy
from app.common.config import TestConfig

sa = SQLAlchemy(config=TestConfig())
# ========================== Schema section begins ==========================


class Mixin:
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(default=func.utc_timestamp())
    updated_at: Mapped[datetime] = mapped_column(
        default=func.utc_timestamp(),
        onupdate=func.utc_timestamp(),
    )
    ip_address: Mapped[Optional[str]] = mapped_column(String(length=40))

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def all_columns(self) -> List[Column]:
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
        session: Optional[AsyncSession] = None,
    ) -> List[Base]:
        return await sa.add_all(cls, *args, autocommit=autocommit, session=session)

    @classmethod
    async def add(
        cls,
        autocommit: bool = False,
        session: Optional[AsyncSession] = None,
        **kwargs: Any,
    ) -> Base:
        return await sa.add(cls, autocommit=autocommit, session=session, **kwargs)

    @classmethod
    async def fetchall_filtered_by(
        cls, session: Optional[AsyncSession] = None, **kwargs: Any
    ) -> List[Base]:
        stmt: Select[Tuple] = select(cls).filter_by(**kwargs)
        return await sa.scalars__fetchall(stmt=stmt, session=session)

    @classmethod
    async def one_filtered_by(
        cls, session: Optional[AsyncSession] = None, **kwargs: Any
    ) -> Base:
        stmt: Select[Tuple] = select(cls).filter_by(**kwargs)
        return await sa.scalars__one(stmt=stmt, session=session)

    @classmethod
    async def fetchall_filtered(
        cls, *criteria: bool, session: Optional[AsyncSession] = None
    ) -> List[Base]:
        stmt: Select[Tuple] = select(cls).filter(*criteria)
        return await sa.scalars__fetchall(stmt=stmt, session=session)

    @classmethod
    async def one_filtered(
        cls, *criteria: bool, session: Optional[AsyncSession] = None
    ) -> Base:
        stmt: Select[Tuple] = select(cls).filter(*criteria)
        return await sa.scalars__one(stmt=stmt, session=session)


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
    access_key: Mapped[str] = mapped_column(String(length=64), index=True, unique=True)
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


class Debugging(Base):
    __tablename__ = "debugging"
    id: Mapped[int] = mapped_column(primary_key=True)


# ========================== Schema section ends ==========================
sa.__pos_init__()


if __name__ == "__main__":
    from asyncio import run

    async def main():
        def log(result: Any, logged_as: str) -> None:
            outputs.append({logged_as: result})

        outputs = []
        random_name = str(uuid4())[:18]
        random_name2 = str(uuid4())[:18]
        random_name3 = str(uuid4())[:18]
        print("\n" * 10)
        try:
            # Create instances
            created_users = await Users.add_all(
                {"username": random_name},
                {"username": random_name2},
                autocommit=True,
            )
            log(
                [created_user.__dict__ for created_user in created_users],
                "created_users",
            )
            created_user = await Users.add(
                autocommit=True, username=random_name3, password=123
            )
            log(created_user.__dict__, "created_user")

            # Query instances
            stmt = select(Users).filter(Users.username.in_([random_name, random_name2]))
            queried_users = await sa.fetchall_scalars(stmt)
            log(
                [queried_user.__dict__ for queried_user in queried_users],
                "queried_users",
            )
            queried_user = await Users.one_filtered_by(
                username=random_name3, password=123
            )
            log(queried_user.__dict__, "queried_user")
            queried_user2 = await Users.fetchall_filtered_by(username=random_name3)
            log(
                [queried_user.__dict__ for queried_user in queried_user2],
                "queried_user2",
            )
        except Exception as e:
            print("<" * 10, "Test failed!", ">" * 10)
            print("Detailed error:\n")
            raise e
        finally:
            await sa.session.close()
            await sa.engine.dispose()
            print("==" * 10, "Outputs", "==" * 10)
            for output in outputs:
                print(output, "\n")

    run(main())
