from dotenv import load_dotenv

load_dotenv()
from collections.abc import Iterable
from asyncio import current_task
from typing import Optional, Any, List, Union, Callable, Type, Tuple
from uuid import uuid4
from urllib import parse
from sqlalchemy import (
    # ScalarResult,
    Result,
    ScalarResult,
    Select,
    create_engine,
    text,
    select,
)
from sqlalchemy.engine.base import Engine, Connection
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
    def __init__(self, config: Union[TestConfig, ProdConfig, LocalConfig]) -> None:
        log(f"Current config status: {config}")
        "{dialect}+{driver}://{user}:{password}@{host}:3306/{database}?charset=utf8mb4"
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
        root_engine = create_engine(root_url, echo=True)
        with root_engine.connect() as conn:
            if not MySQL.is_user_exists(config.mysql_user, engine_or_conn=conn):
                MySQL.create_user(
                    config.mysql_user, config.mysql_password, "%", engine_or_conn=conn
                )
            if not MySQL.is_db_exists(config.mysql_database, engine_or_conn=conn):
                MySQL.create_db(config.mysql_database, engine_or_conn=conn)
            if not MySQL.is_user_granted(
                config.mysql_user, config.mysql_database, engine_or_conn=conn
            ):
                MySQL.grant_user(
                    "ALL PRIVILEGES",
                    "testing_db.*",
                    config.mysql_user,
                    "%",
                    engine_or_conn=conn,
                )
            Base.metadata.drop_all(conn) if config.test_mode else ...
            Base.metadata.create_all(conn)
            conn.commit()
        root_engine.dispose()
        self.engine = create_async_engine(
            database_url,
            echo=config.db_echo,
            pool_recycle=config.db_pool_recycle,
            pool_pre_ping=True,
        )
        print(database_url)
        self.session = async_scoped_session(
            async_sessionmaker(
                bind=self.engine, autocommit=False, autoflush=False, future=True
            ),
            scopefunc=current_task,
        )

    async def get_db(self) -> AsyncSession:
        async with self.session() as transaction:
            yield transaction

    def check_connectivity(self, root_engine: Engine, database_name: str) -> None:
        is_assertion_error_occured = False
        while True:  # Connection check
            try:
                assert MySQL.is_db_exists(root_engine, database=database_name)
            except AssertionError:
                if is_assertion_error_occured:
                    raise Exception("Infinite-looping error")
                is_assertion_error_occured = True
                print(f"Database {database_name} not exists. Creating new database...")
                MySQL.create_db(root_engine, database=database_name)
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
                async with self.session() as transaction:
                    result = await _func(transaction, *args, **kwargs)
                    print(result)
                    await transaction.commit() if autocommit else ...
            else:
                result = await _func(session, *args, **kwargs)
                await session.commit() if autocommit else ...
            return result

        return _wrapper


sa = SQLAlchemy(config=TestConfig())


class SaBeginMethods:
    @staticmethod
    @sa.run_in_session
    async def execute(session: AsyncSession, stmt: Union[text, Select]) -> Result:
        return await session.execute(stmt)

    @staticmethod
    @sa.run_in_session
    async def scalar(session: AsyncSession, stmt: Select) -> Any:
        return await session.scalar(stmt)

    @staticmethod
    @sa.run_in_session
    async def scalars(session: AsyncSession, stmt: Select) -> ScalarResult:
        return await session.scalars(stmt)

    @staticmethod
    @sa.run_in_session
    async def add(session: AsyncSession, instance: Base) -> Base:
        session.add(instance)
        return instance

    @staticmethod
    @sa.run_in_session
    async def add_all(
        session: AsyncSession, instances: Iterable[Base]
    ) -> Iterable[Base]:
        session.add_all(instances)
        return instances


class SaNestedMethods:
    @staticmethod
    async def scalars__fetchall(
        stmt: Select, session: Optional[AsyncSession] = None
    ) -> List[Base]:
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

    @staticmethod
    async def scalars__one_or_none(
        stmt: Select, session: Optional[AsyncSession] = None
    ) -> Optional[Base]:
        return (
            await SaBeginMethods.scalars(session=session, autocommit=False, stmt=stmt)
        ).one_or_none()


# Default methods =======================================================


async def fetchall_scalars(
    stmt: Select, session: Optional[AsyncSession] = None
) -> List[Base]:
    return await SaNestedMethods.scalars__fetchall(stmt=stmt, session=session)


async def one_scalars(stmt: Select, session: Optional[AsyncSession] = None) -> Base:
    return await SaNestedMethods.scalars__one(stmt=stmt, session=session)


async def one_or_none_scalars(
    stmt: Select, session: Optional[AsyncSession] = None
) -> Optional[Base]:
    return await SaNestedMethods.scalars__one_or_none(stmt=stmt, session=session)


async def add_all(
    schema: Type[Base],
    *args: dict,
    autocommit: bool = False,
    session: Optional[AsyncSession] = None,
) -> List[Base]:
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
) -> Base:
    instance = schema(**kwargs)
    await SaBeginMethods.add(session=session, autocommit=autocommit, instance=instance)
    return instance


# Other custom methods =======================================================


async def fetchall_filtered_by(
    schema: Type[Base], session: Optional[AsyncSession] = None, **kwargs: Any
) -> List[Base]:
    stmt: Select[Tuple] = select(schema).filter_by(**kwargs)
    return await SaNestedMethods.scalars__fetchall(stmt=stmt, session=session)


async def one_filtered_by(
    schema: Type[Base], session: Optional[AsyncSession] = None, **kwargs: Any
) -> Base:
    stmt: Select[Tuple] = select(schema).filter_by(**kwargs)
    return await SaNestedMethods.scalars__one(stmt=stmt, session=session)


async def fetchall_filtered(
    schema: Type[Base], *criteria: bool, session: Optional[AsyncSession] = None
) -> List[Base]:
    stmt: Select[Tuple] = select(schema).filter(*criteria)
    return await SaNestedMethods.scalars__fetchall(stmt=stmt, session=session)


async def one_filtered(
    schema: Type[Base], *criteria: bool, session: Optional[AsyncSession] = None
) -> Base:
    stmt: Select[Tuple] = select(schema).filter(*criteria)
    return await SaNestedMethods.scalars__one(stmt=stmt, session=session)


# Test section =======================================================


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
        # queried_users = await fetchall_filtered(
        #     Users, Users.username.in_([random_name, random_name2])
        # )
        stmt = select(Users).filter(Users.username.in_([random_name, random_name2]))
        queried_users = await fetchall_scalars(stmt)
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
            print(output, "\n")


if __name__ == "__main__":
    from asyncio import run

    run(main())
