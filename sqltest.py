from asyncio import current_task
from typing import Optional, Any, List, Union
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ResourceClosedError
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    async_scoped_session,
    create_async_engine,
    AsyncSession,
)
from sqlalchemy.orm import declarative_base
from time import sleep
import logging
from datetime import datetime
from app.common.config import TestConfig, ProdConfig, LocalConfig

Base = declarative_base()


def log(msg) -> None:
    logging.critical(f"[{datetime.now()}] {msg}")


class MySQL:
    @staticmethod
    def get_query(engine: Engine, query: str) -> Optional[Any]:
        with engine.connect() as conn:
            result = conn.execute(
                text(query + ";" if not query.endswith(";") else query)
            )
            return result.all()

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
            cls.get_query(
                engine=engine,
                query=f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{database_name}';",
            )
        )

    @classmethod
    def drop_db(cls, engine: Engine, database_name: str) -> None:
        return cls.get_query(
            engine=engine,
            query=f"DROP DATABASE {database_name};",
        )

    @classmethod
    def create_db(cls, engine: Engine, database_name: str) -> None:
        return cls.get_query(
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
        return cls.get_query(
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
        return cls.get_query(
            engine=engine,
            query=f"GRANT {grant} ON {on} TO '{to_user}'@'{user_host}'",
        )


class SQLAlchemy:
    def __init__(
        self,
    ):
        self.engine: Optional[Engine] = None
        self.session: AsyncSession = None

    def init_db(self, config: Union[TestConfig, ProdConfig, LocalConfig]) -> None:
        log(f"Current config status: {config}")
        root_engine = create_engine(
            config.mysql_root_url.replace("aiomysql", "pymysql"),
            echo=config.db_echo,
        )
        self.check_connectivity(
            root_engine=root_engine, database_name=config.mysql_database
        )
        root_engine.dispose()
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


if __name__ == "__main__":
    sql_alchemy = SQLAlchemy()
    sql_alchemy.init_db(config=TestConfig())
