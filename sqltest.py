from dotenv import load_dotenv

load_dotenv()
from collections.abc import Iterable
from asyncio import current_task
from typing import Optional, Any, List, Union, Callable, Type
from uuid import uuid4
from sqlalchemy import (
    ScalarResult,
    Select,
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

    def run_in_session(self, _func: Callable[[AsyncSession, bool, list, dict], Any]):
        async def _wrapper(
            session: Optional[AsyncSession] = None,
            autocommit: bool = False,
            *args: list,
            **kwargs,
        ):
            if session is None:
                async with self.session() as tmp_session:
                    result = await _func(tmp_session, *args, **kwargs)
                    await tmp_session.commit() if autocommit else ...
            else:
                result = await _func(session, *args, **kwargs)
                await session.commit() if autocommit else ...
            return result

        return _wrapper


sa = SQLAlchemy(config=TestConfig())


class SaBeginMethods:
    @staticmethod
    @sa.run_in_session
    async def execute(session: AsyncSession, stmt: Union[text, Select]):
        return await session.execute(stmt)

    @staticmethod
    @sa.run_in_session
    async def scalar(session: AsyncSession, stmt: Select):
        return await session.scalar(stmt)

    @staticmethod
    @sa.run_in_session
    async def scalars(session: AsyncSession, stmt: Select):
        return await session.scalars(stmt)

    @staticmethod
    @sa.run_in_session
    async def add(session: AsyncSession, instance: Base):
        session.add(instance)
        return instance

    @staticmethod
    @sa.run_in_session
    async def add_all(session: AsyncSession, instances: Iterable[Base]):
        session.add_all(instances)
        return instances


class SaNestedMethods:
    @staticmethod
    async def scalars__fetchall(
        stmt: Select, session: Optional[AsyncSession] = None
    ) -> List[Base]:
        # (await session.ad(stmt)).fetchall
        return (
            await SaBeginMethods.scalars(session=session, autocommit=False, stmt=stmt)
        ).fetchall()

    @staticmethod
    async def scalars__one(
        stmt: Select, session: Optional[AsyncSession] = None
    ) -> Base:
        return (
            await SaBeginMethods.scalars(session=session, autocommit=False, stmt=stmt)
        ).one()


async def add_all(
    schema: Type[Base],
    *args: dict,
    autocommit: bool = False,
    session: Optional[AsyncSession] = None,
):
    instances = [schema(**arg) for arg in args]
    await SaBeginMethods.add_all(
        session=session, autocommit=autocommit, instances=instances
    )
    return instances


async def add(
    schema: Type[Base],
    autocommit: bool = False,
    session: Optional[AsyncSession] = None,
    **kwargs: Any,
):
    instance = schema(**kwargs)
    await SaBeginMethods.add(session=session, autocommit=autocommit, instance=instance)
    return instance


async def fetchall_filtered_by(
    schema: Type[Base], session: Optional[AsyncSession] = None, **kwargs: Any
):
    stmt = select(schema).filter_by(**kwargs)
    return await SaNestedMethods.scalars__fetchall(stmt=stmt, session=session)


async def one_filtered_by(
    schema: Type[Base], session: Optional[AsyncSession] = None, **kwargs: Any
):
    stmt = select(schema).filter_by(**kwargs)
    return await SaNestedMethods.scalars__one(stmt=stmt, session=session)


async def fetchall_filtered(
    schema: Type[Base], *criteria: Any, session: Optional[AsyncSession] = None
):
    stmt = select(schema).filter(*criteria)
    return await SaNestedMethods.scalars__fetchall(stmt=stmt, session=session)


async def one_filtered(
    schema: Type[Base], *criteria: Any, session: Optional[AsyncSession] = None
):
    stmt = select(schema).filter(*criteria)
    return await SaNestedMethods.scalars__one(stmt=stmt, session=session)


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
        created_users = await add_all(
            Users,
            {"username": random_name},
            {"username": random_name2},
            autocommit=True,
        )
        log([created_user.__dict__ for created_user in created_users], "created_users")
        created_user = await add(
            Users, autocommit=True, username=random_name3, password=123
        )
        log(created_user.__dict__, "created_user")

        # Find instances
        queried_users = await fetchall_filtered(
            Users, Users.username.in_([random_name, random_name2])
        )
        log([queried_user.__dict__ for queried_user in queried_users], "queried_users")

        queried_user = await one_filtered_by(Users, username=random_name3, password=123)
        log(queried_user.__dict__, "queried_user")
        queried_user2 = await fetchall_filtered_by(Users, username=random_name3)
        log([queried_user.__dict__ for queried_user in queried_user2], "queried_user2")
        await sa.session.close()
        await sa.engine.dispose()
    except Exception as e:

        print("<" * 10, "Test failed!", ">" * 10)
        print("Detailed error:\n", e)
    finally:
        print("==" * 10, "Outputs", "==" * 10)
        for output in outputs:
            print(output, '\n')


if __name__ == "__main__":
    from asyncio import run

    run(main())
