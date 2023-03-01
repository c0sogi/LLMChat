from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    func,
    Enum,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import Session, relationship, Query
from app.database.connection import Base, db
from app.errors.exceptions import SqlFailureEx
from typing import List, Optional, Union


class CustomMixin:
    __table__ = None
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, nullable=False, default=func.utc_timestamp())
    updated_at = Column(
        DateTime,
        nullable=False,
        default=func.utc_timestamp(),
        onupdate=func.utc_timestamp(),
    )
    ip_address = Column(String(length=40), nullable=True)

    def __init__(self):
        self._queries: Base = None
        self._session: Optional[Session] = None

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
    async def create_schema_instance(
        cls, session: Session, auto_commit: bool = False, **kwargs
    ) -> Base:
        """
        Create data in DB and return this new DB model instance.
        """
        db_model_instance = cls()
        with session.begin_nested():
            try:
                for column in db_model_instance.all_columns:
                    setattr(
                        db_model_instance, column.name, kwargs.get(column.name)
                    ) if column.name in kwargs else None
                session.add(db_model_instance)
                session.flush()
            except Exception as e:
                print(e)
                raise SqlFailureEx
            else:
                session.commit() if auto_commit else ...
        return db_model_instance

    @classmethod
    async def get_row_from_db(
        cls, session: Optional[Session] = None, **kwargs
    ) -> Optional[Query]:
        """
        Get matched query(row) from DB
        """
        is_session_given = False if session is None else True
        session = db.session_local() if session is None else session
        try:
            query = session.query(cls).filter(
                *[getattr(cls, col) == val for col, val in kwargs.items()]
            )
            assert (
                query.count() <= 1
            ), "Only one row is supposed to be returned, but got more than one."
            return query.first()
        except Exception as e:
            print(e)
            raise SqlFailureEx
        finally:
            session.close() if not is_session_given else ...

    @classmethod
    async def get_filtered_schema_instances(cls, session: Session, **kwargs) -> Base:
        """
        Return a DB model instance with filtered queries. Those queries can be accessed by "._queries"
        """
        session = db.session_local() if session is None else session
        try:
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
                    conditions[filtering_condition](
                        column_to_be_filtered, filtering_value
                    )
                )
            db_model_instance = cls()
            db_model_instance._session = session
            db_model_instance._queries = db_model_instance._session.query(cls).filter(
                *conditions_to_query
            )
            return db_model_instance
        except Exception as e:
            print(e)

    @classmethod
    def get_column(cls, column_name: str) -> Column:
        return getattr(cls, column_name)

    @classmethod
    def get_db_model(cls) -> Base:
        return cls

    def order_by(self, *args: str) -> Base:
        for arg in args:
            is_ascending_order = False if arg.startswith("-") else True
            column_name = arg[1:] if arg.startswith("-") else arg
            column = self.get_column(column_name)
            self._queries = (
                self._queries.order_by(column.asc())
                if is_ascending_order
                else self._queries.order_by(column.desc())
            )
        return self

    def update(self, auto_commit: bool = False, **kwargs) -> Union[Query, None]:
        updated_queries_number = self._queries.update(kwargs)
        self._session.flush()
        self._session.commit() if auto_commit else ...
        return self._queries.first() if updated_queries_number > 0 else None

    def first(self) -> Union[Query, None]:
        return self._queries.first()

    def delete(self, auto_commit: bool = False) -> None:
        self._queries.delete()
        self._session.commit() if auto_commit else ...

    def all(self) -> list:
        return self._queries.all()

    def count(self) -> int:
        return self._queries.count()

    def close(self) -> None:
        self._session.close()


class Users(Base, CustomMixin):
    __tablename__ = "users"
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
