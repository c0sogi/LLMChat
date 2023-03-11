# from dotenv import load_dotenv
# if environ.get("API_ENV") is None:
#     load_dotenv()
#     environ.update({"API_ENV": "test"})
from collections.abc import Iterable
from asyncio import current_task
from typing import Optional, Any, List, Union, Callable, Type
from urllib import parse
from sqlalchemy import (
    Result,
    ScalarResult,
    Select,
    Delete,
    Update,
    create_engine,
    text,
)
from sqlalchemy.engine.base import Engine, Connection
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    async_scoped_session,
    create_async_engine,
    AsyncSession,
    AsyncEngine,
)
from sqlalchemy_utils import database_exists, create_database
import logging
from datetime import datetime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm.decl_api import DeclarativeMeta
from app.common.config import TestConfig, ProdConfig, LocalConfig, SingletonMetaClass

Base: DeclarativeMeta = declarative_base()


class MySQL:
    query_set: dict = {
        "is_user_exists": "SELECT EXISTS(SELECT 1 FROM mysql.user WHERE user = '{user}');",
        "is_db_exists": "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{database}';",
        "is_user_granted": (
            "SELECT * FROM information_schema.schema_privileges "
            "WHERE table_schema = '{database}' AND grantee = '{user}';"
        ),
        "create_user": "CREATE USER '{user}'@'{host}' IDENTIFIED BY '{password}'",
        "grant_user": "GRANT {grant} ON {on} TO '{to_user}'@'{user_host}'",
        "create_db": "CREATE DATABASE {database} CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;",
        "drop_db": "DROP DATABASE {database};",
    }

    @staticmethod
    def execute(
        query: str, engine_or_conn: Union[Engine, Connection], scalar: bool = False
    ) -> Optional[Any]:
        if isinstance(engine_or_conn, Engine) and not isinstance(
            engine_or_conn, Connection
        ):
            with engine_or_conn.connect() as conn:
                cursor = conn.execute(
                    text(query + ";" if not query.endswith(";") else query)
                )
                return cursor.scalar() if scalar else None
        elif isinstance(engine_or_conn, Connection):
            cursor = engine_or_conn.execute(
                text(query + ";" if not query.endswith(";") else query)
            )
            return cursor.scalar() if scalar else None

    @staticmethod
    def clear_all_table_data(engine: Engine, except_tables: Optional[List[str]] = None):
        with engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            for table in Base.metadata.sorted_tables:
                conn.execute(table.delete()) if table.name not in except_tables else ...
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            conn.commit()

    @classmethod
    def is_db_exists(
        cls, database: str, engine_or_conn: Union[Engine, Connection]
    ) -> bool:
        return bool(
            cls.execute(
                cls.query_set["is_db_exists"].format(database=database),
                engine_or_conn,
                scalar=True,
            )
        )

    @classmethod
    def is_user_exists(
        cls, user: str, engine_or_conn: Union[Engine, Connection]
    ) -> bool:
        return bool(
            cls.execute(
                cls.query_set["is_user_exists"].format(user=user),
                engine_or_conn,
                scalar=True,
            )
        )

    @classmethod
    def is_user_granted(
        cls, user: str, database: str, engine_or_conn: Union[Engine, Connection]
    ) -> bool:
        return bool(
            cls.execute(
                cls.query_set["is_user_granted"].format(user=user, database=database),
                engine_or_conn,
                scalar=True,
            )
        )

    @classmethod
    def drop_db(cls, database: str, engine_or_conn: Union[Engine, Connection]) -> None:
        return cls.execute(
            cls.query_set["drop_db"].format(database=database),
            engine_or_conn,
        )

    @classmethod
    def create_db(
        cls, database: str, engine_or_conn: Union[Engine, Connection]
    ) -> None:
        return cls.execute(
            cls.query_set["create_db"].format(database=database),
            engine_or_conn,
        )

    @classmethod
    def create_user(
        cls,
        user: str,
        password: str,
        host: str,
        engine_or_conn: Union[Engine, Connection],
    ) -> None:
        return cls.execute(
            cls.query_set["create_user"].format(
                user=user, password=password, host=host
            ),
            engine_or_conn,
        )

    @classmethod
    def grant_user(
        cls,
        grant: str,
        on: str,
        to_user: str,
        user_host: str,
        engine_or_conn: Union[Engine, Connection],
    ) -> None:
        return cls.execute(
            cls.query_set["grant_user"].format(
                grant=grant, on=on, to_user=to_user, user_host=user_host
            ),
            engine_or_conn,
        )


class SQLAlchemy(metaclass=SingletonMetaClass):
    def __init__(self):
        self.is_test_mode: bool = None
        self.root_engine: Engine = None
        self.engine: AsyncEngine = None
        self.session: AsyncSession = None
        self.is_initiated = False

    def init(self, config: Union[TestConfig, ProdConfig, LocalConfig]) -> None:
        if self.is_initiated:
            return
        self.is_test_mode = True if config.test_mode else False
        SQLAlchemy.log(f"Current config status: {config}")
        root_url = config.database_url_format.format(
            dialect="mysql",
            driver="pymysql",
            user="root",
            password=parse.quote(config.mysql_root_password),
            host=config.mysql_host,
            database=config.mysql_database,
        )
        database_url = config.database_url_format.format(
            dialect="mysql",
            driver="aiomysql",
            user=config.mysql_user,
            password=parse.quote(config.mysql_password),
            host=config.mysql_host,
            database=config.mysql_database,
        )
        if not database_exists(root_url):
            create_database(root_url)

        self.root_engine = create_engine(root_url, echo=True if config.debug else False)
        with self.root_engine.connect() as conn:
            if not MySQL.is_user_exists(config.mysql_user, engine_or_conn=conn):
                MySQL.create_user(
                    config.mysql_user, config.mysql_password, "%", engine_or_conn=conn
                )
            if not MySQL.is_user_granted(
                config.mysql_user, config.mysql_database, engine_or_conn=conn
            ):
                MySQL.grant_user(
                    "ALL PRIVILEGES",
                    f"{config.mysql_database}.*",
                    config.mysql_user,
                    "%",
                    engine_or_conn=conn,
                )
            Base.metadata.drop_all(conn) if self.is_test_mode else ...
            Base.metadata.create_all(conn)
            conn.commit()
        self.root_engine.dispose()
        self.engine = create_async_engine(
            database_url,
            echo=config.db_echo,
            pool_recycle=config.db_pool_recycle,
            pool_pre_ping=True,
        )
        self.session = async_scoped_session(
            async_sessionmaker(
                bind=self.engine, autocommit=False, autoflush=False, future=True
            ),
            scopefunc=current_task,
        )
        self.is_initiated = True

    async def get_db(self) -> AsyncSession:
        async with self.session() as transaction:
            yield transaction

    def run_in_session(self, func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(
            session: Optional[AsyncSession] = None,
            autocommit: bool = False,
            refresh: bool = False,
            *args: Any,
            **kwargs: Any,
        ):
            if session is None:
                async with self.session() as transaction:
                    result = await func(transaction, *args, **kwargs)
                    if autocommit:
                        await transaction.commit()
                    if refresh:
                        [await transaction.refresh(r) for r in result] if isinstance(
                            result, Iterable
                        ) else await transaction.refresh(result)
            else:
                result = await func(session, *args, **kwargs)
                if autocommit:
                    await session.commit()
                if refresh:
                    [await session.refresh(r) for r in result] if isinstance(
                        result, Iterable
                    ) else await session.refresh(result)
            return result

        return wrapper

    @staticmethod
    def log(msg) -> None:
        logging.critical(f"[{datetime.now()}] {msg}")

    async def _execute(  # To be decorated
        self, session: AsyncSession, stmt: Union[text, Update, Delete, Select]
    ) -> Result:
        return await session.execute(stmt)

    async def _scalar(  # To be decorated
        self,
        session: AsyncSession,
        stmt: Select,
    ) -> Any:
        return await session.scalar(stmt)

    async def _scalars(  # To be decorated
        self,
        session: AsyncSession,
        stmt: Select,
    ) -> ScalarResult:
        return await session.scalars(stmt)

    async def _add(  # To be decorated
        self,
        session: AsyncSession,
        instance: Base,
    ) -> Base:
        session.add(instance)
        return instance

    async def _add_all(  # To be decorated
        self,
        session: AsyncSession,
        instances: Iterable[Base],
    ) -> Iterable[Base]:
        session.add_all(instances)
        return instances

    async def _delete(  # To be decorated
        self,
        session: AsyncSession,
        instance: Base,
    ) -> Base:
        await session.delete(instance)
        return instance

    async def execute(
        self,
        stmt: Union[text, Update, Delete, Select],
        autocommit: bool = False,
        refresh: bool = False,
        session: Optional[AsyncSession] = None,
    ) -> Result:
        return await self.run_in_session(self._execute)(
            session, autocommit=autocommit, refresh=refresh, stmt=stmt
        )

    async def scalar(self, stmt: Select, session: Optional[AsyncSession] = None) -> Any:
        return await self.run_in_session(self._scalar)(session, stmt=stmt)

    async def scalars(
        self, stmt: Select, session: Optional[AsyncSession] = None
    ) -> ScalarResult:
        return await self.run_in_session(self._scalars)(session, stmt=stmt)

    async def add(
        self,
        schema: Type[Base],
        autocommit: bool = False,
        refresh: bool = False,
        session: Optional[AsyncSession] = None,
        **kwargs: Any,
    ) -> Base:
        instance = schema(**kwargs)
        return await self.run_in_session(self._add)(
            session, autocommit=autocommit, refresh=refresh, instance=instance
        )

    async def add_all(
        self,
        schema: Type[Base],
        *args: dict,
        autocommit: bool = False,
        refresh: bool = False,
        session: Optional[AsyncSession] = None,
    ) -> List[Base]:
        instances = [schema(**arg) for arg in args]
        return await self.run_in_session(self._add_all)(
            session, autocommit=autocommit, refresh=refresh, instances=instances
        )

    async def delete(
        self,
        instance: Base,
        autocommit: bool = False,
        session: Optional[AsyncSession] = None,
    ) -> Base:
        return await self.run_in_session(self._delete)(
            session, autocommit=autocommit, instance=instance
        )

    async def scalars__fetchall(
        self, stmt: Select, session: Optional[AsyncSession] = None
    ) -> List[Base]:
        return (await self.run_in_session(self._scalars)(session, stmt=stmt)).fetchall()

    async def scalars__one(
        self, stmt: Select, session: Optional[AsyncSession] = None
    ) -> Base:
        return (await self.run_in_session(self._scalars)(session, stmt=stmt)).one()

    async def scalars__first(
        self, stmt: Select, session: Optional[AsyncSession] = None
    ) -> Base:
        return (await self.run_in_session(self._scalars)(session, stmt=stmt)).first()

    async def scalars__one_or_none(
        self, stmt: Select, session: Optional[AsyncSession] = None
    ) -> Optional[Base]:
        return (
            await self.run_in_session(self._scalars)(session, stmt=stmt)
        ).one_or_none()
