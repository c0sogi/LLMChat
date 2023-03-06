from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    func,
    Enum,
    Boolean,
    ForeignKey,
    select,
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import Base, db
from typing import List, Optional


class CustomMixin:
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, nullable=False, default=func.utc_timestamp())
    updated_at = Column(
        DateTime,
        nullable=False,
        default=func.utc_timestamp(),
        onupdate=func.utc_timestamp(),
    )
    ip_address = Column(String(length=40), nullable=True)

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
    async def create_new(
        cls, session: Optional[AsyncSession] = None, auto_commit: bool = False, **kwargs
    ) -> Base:
        """
        Create data in DB and return this new DB model instance.
        """
        instance = cls()
        for column in instance.all_columns:
            setattr(
                instance, column.name, kwargs.get(column.name)
            ) if column.name in kwargs else None
        if session is None:
            async with db.session() as session:
                session.add(instance)
                await session.commit() if auto_commit else ...
                await session.refresh(instance) if auto_commit else ...
        else:
            session.add(instance)
            await session.commit() if auto_commit else ...
            await session.refresh(instance) if auto_commit else ...
        return instance

    @classmethod
    async def filter_by_equality(cls, session: Optional[AsyncSession] = None, **kwargs):
        query = select(cls).filter_by(**kwargs)
        if session is None:
            async with db.session() as session:
                query_result = await session.execute(query)
                return query_result.scalars()
        else:
            return (await session.execute(query)).scalars()

    @classmethod
    async def filter_by_condition(
        cls, session: Optional[AsyncSession] = None, **kwargs
    ):
        """
        Return a DB model instance with filtered queries. Those queries can be accessed by "._queries"
        """
        conditions = {
            "in": lambda col, val: col.in_(val),
            "eq": lambda col, val: col == val,
            "gt": lambda col, val: col > val,
            "gte": lambda col, val: col >= val,
            "lt": lambda col, val: col < val,
            "lte": lambda col, val: col <= val,
        }
        conditions_to_query = []
        for filtering_statement, filtering_value in kwargs.items():
            filtering_statement = filtering_statement.split("__")
            assert len(filtering_statement) <= 2, "No 2 more double underscores"
            filtering_condition = (
                filtering_statement[1] if len(filtering_statement) == 2 else "eq"
            )
            column_to_be_filtered = getattr(cls, filtering_statement[0])
            conditions_to_query.append(
                conditions[filtering_condition](column_to_be_filtered, filtering_value)
            )
        query = select(cls).filter(*conditions_to_query)
        if session is None:
            async with db.session() as session:
                return (await session.execute(query)).scalars()
        else:
            return (await session.execute(query)).scalars()

    @classmethod
    def get_column(cls, column_name: str) -> Column:
        return getattr(cls, column_name)

    @classmethod
    def get_class(cls) -> Base:
        return cls

    @classmethod
    async def order_by(cls, session: Optional[AsyncSession] = None, *args: str):
        query = select(cls)
        for arg in args:
            is_ascending_order = False if arg.startswith("-") else True
            column_name = arg[1:] if arg.startswith("-") else arg
            column = cls.get_column(column_name)
            query = (
                query.order_by(column.asc())
                if is_ascending_order
                else query.order_by(column.desc())
            )
        if session is None:
            async with db.session() as session:
                return await session.execute(query).scalars()
        else:
            return await session.execute(query).scalars()

    async def update(
        self,
        session: Optional[AsyncSession] = None,
        auto_commit: bool = False,
        **kwargs: str
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if session is None:
            async with db.session() as session:
                await session.commit() if auto_commit else ...
                await session.refresh(self) if auto_commit else ...
        else:
            await session.commit() if auto_commit else ...
            await session.refresh(self) if auto_commit else ...
        return self


class Users(CustomMixin, Base):
    __tablename__ = "users"
    # id = Column(Integer, primary_key=True, index=True)
    status = Column(Enum("active", "deleted", "blocked"), default="active")
    email = Column(String(length=255), nullable=True)
    pw = Column(String(length=2000), nullable=True)
    name = Column(String(length=255), nullable=True)
    phone_number = Column(String(length=20), nullable=True, unique=True)
    profile_img = Column(String(length=1000), nullable=True)
    sns_type = Column(Enum("FB", "G", "K"), nullable=True)
    marketing_agree = Column(Boolean, nullable=True, default=True)
    keys = relationship("ApiKeys", back_populates="users")


class ApiKeys(Base, CustomMixin):
    __tablename__ = "api_keys"
    access_key = Column(String(length=64), nullable=False, index=True)
    secret_key = Column(String(length=64), nullable=False)
    user_memo = Column(String(length=40), nullable=True)
    status = Column(Enum("active", "stopped", "deleted"), default="active")
    is_whitelisted = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    whitelist = relationship("ApiWhiteLists", backref="api_keys")
    users = relationship("Users", back_populates="keys")


class ApiWhiteLists(Base, CustomMixin):
    __tablename__ = "api_whitelists"
    ip_addr = Column(String(length=64), nullable=False)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
