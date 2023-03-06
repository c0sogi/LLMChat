from dotenv import load_dotenv

load_dotenv()
from collections.abc import Iterable
from asyncio import current_task
from typing import Optional, Any, List, Union, Callable
from uuid import uuid4
from sqlalchemy import (
    ScalarResult,
    create_engine,
    text,
    select,
)
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    async_scoped_session,
    create_async_engine,
    AsyncSession,
)
from time import sleep
import logging
from datetime import datetime
from app.common.config import TestConfig, ProdConfig, LocalConfig, SingletonMetaClass
from sqltestschema import Base, Users


def log(msg) -> None:
    logging.critical(f"[{datetime.now()}] {msg}")


class MySQL:
    @staticmethod
    def execute(engine: Engine, query: str) -> Optional[Any]:
        with engine.connect() as conn:
            return conn.execute(text(query + ";" if not query.endswith(";") else query))

    @staticmethod
    def clear_all_table_data(engine: Engine, except_tables: Optional[List[str]] = None):
        with engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            for table in Base.metadata.sorted_tables:
                conn.execute(table.delete()) if table.name not in except_tables else ...
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            conn.commit()

    @classmethod
    def is_db_exists(cls, engine: Engine, database_name: str) -> bool:
        return bool(
            cls.execute(
                engine=engine,
                query=f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{database_name}';",
            )
        )

    @classmethod
    def drop_db(cls, engine: Engine, database_name: str) -> None:
        return cls.execute(
            engine=engine,
            query=f"DROP DATABASE {database_name};",
        )

    @classmethod
    def create_db(cls, engine: Engine, database_name: str) -> None:
        return cls.execute(
            engine=engine,
            query=f"CREATE DATABASE {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;",
        )

    @classmethod
    def create_user(
        cls,
        engine: Engine,
        username: str,
        password: str,
        host: str,
    ) -> None:
        return cls.execute(
            engine=engine,
            query=f"CREATE USER '{username}'@'{host}' IDENTIFIED BY '{password}'",
        )

    @classmethod
    def grant_user(
        cls,
        engine: Engine,
        grant: str,
        on: str,
        to_user: str,
        user_host: str,
    ) -> None:
        return cls.execute(
            engine=engine,
            query=f"GRANT {grant} ON {on} TO '{to_user}'@'{user_host}'",
        )


class SQLAlchemy(metaclass=SingletonMetaClass):
    def __init__(self, config: Union[TestConfig, ProdConfig, LocalConfig]) -> None:
        log(f"Current config status: {config}")
        # root_engine = create_engine(
        #     config.mysql_root_url.replace("aiomysql", "pymysql"),
        #     echo=config.db_echo,
        # )
        # self.check_connectivity(
        #     root_engine=root_engine, database_name=config.mysql_database
        # )
        # root_engine.dispose()
        engine = create_engine(
            config.mysql_url.replace("aiomysql", "pymysql"),
            echo=config.db_echo,
        )
        with engine.connect() as conn:
            Base.metadata.drop_all(conn) if config.test_mode else ...
            Base.metadata.create_all(conn)
        engine.dispose()
        self.engine = create_async_engine(
            config.mysql_url,
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

    async def get_db(self) -> AsyncSession:
        async with self.session() as session:
            yield session

    def check_connectivity(self, root_engine: Engine, database_name: str) -> None:
        is_assertion_error_occured = False
        while True:  # Connection check
            try:
                assert MySQL.is_db_exists(root_engine, database_name=database_name)
            except AssertionError:
                if is_assertion_error_occured:
                    raise Exception("Infinite-looping error")
                is_assertion_error_occured = True
                print(f"Database {database_name} not exists. Creating new database...")
                MySQL.create_db(root_engine, database_name=database_name)
            except Exception as e:
                print(e)
                sleep(5)
            else:
                return

    def _run_in_session(self, _func: Callable[[AsyncSession, dict], Any]):
        async def _wrapper(
            schema: Base,
            session: Optional[AsyncSession] = None,
            autocommit: bool = False,
            stmt: Any = None,
            **kwargs,
        ):
            if session is None:
                async with self.session() as tmp_session:
                    result = await _func(schema, tmp_session, stmt, **kwargs)
                    await tmp_session.commit() if autocommit else ...
            else:
                result = await _func(schema, session, stmt, **kwargs)
                await session.commit() if autocommit else ...
            return result

        return _wrapper


sa = SQLAlchemy(config=TestConfig())


@sa._run_in_session
async def add_by(
    schema: Base, session: Optional[AsyncSession] = None, stmt: dict = None
):
    if stmt is not None:
        instance = schema(**stmt)
        session.add(instance)
        return instance


@sa._run_in_session
async def add_by_list(
    schema: Base,
    session: Optional[AsyncSession] = None,
    stmt: Iterable[dict] = None,
):
    if stmt is not None:
        instances = [schema(**attributes) for attributes in stmt]
        session.add_all(instances)
        return instances


@sa._run_in_session
async def where_all(
    schema: Base, session: Optional[AsyncSession] = None, stmt: bool = None
) -> List[Base]:
    return (
        [result for result in await session.scalars(select(schema).where(stmt))]
        if stmt is not None
        else None
    )


@sa._run_in_session
async def where_first(
    schema: Base, session: Optional[AsyncSession] = None, stmt: bool = None
) -> Optional[Base]:
    return (
        (await session.scalars(select(schema).where(stmt))).first()
        if stmt is not None
        else None
    )


@sa._run_in_session
async def get_by_primary_key(
    schema: Base, session: Optional[AsyncSession] = None, stmt: bool = None
) -> Optional[Base]:
    session.get()
    return (
        (await session.scalars(select(schema).where(stmt))).first()
        if stmt is not None
        else None
    )


async def main():
    random_name = str(uuid4())[:18]
    random_name2 = str(uuid4())[:18]
    random_name3 = str(uuid4())[:18]
    print("\n" * 10)

    # Create instances
    two_users = await add_by_list(
        Users,
        autocommit=True,
        stmt=[{"username": random_name}, {"username": random_name2}],
    )
    one_user = await add_by(
        Users,
        autocommit=True,
        stmt={"username": random_name3, "password": 123},
    )

    # Find instances
    two_users = await where_all(
        Users, stmt=Users.username.in_([random_name, random_name2])
    )
    for user in two_users:
        print("Queried user:", user.__dict__)
    one_user = await where_first(
        Users, stmt=(Users.username == random_name3 and Users.password == 123)
    )
    print("Queried another user:", one_user)
    await sa.session.close()
    await sa.engine.dispose()


if __name__ == "__main__":
    from asyncio import run

    run(main())
