from asyncio import current_task, run
from typing import Union, Optional, Any, List
from fastapi import FastAPI
from sqlalchemy import text, create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import OperationalError, ResourceClosedError
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    async_scoped_session,
    create_async_engine,
    AsyncSession,
)
from sqlalchemy.orm import declarative_base
from time import sleep
import logging
from app.common.config import TestConfig, ProdConfig, LocalConfig

Base = declarative_base()


class MySQL:
    @staticmethod
    def get_query(engine: Engine, query: str) -> Optional[Any]:
        with engine.connect() as conn:
            result = conn.execute(
                text(query + ";" if not query.endswith(";") else query)
            )
            try:
                result = result.fetchall()
            except ResourceClosedError:
                result = None
            print(f">>> Query '{query}' result:", result)
            return result

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
        app: FastAPI = None,
        config: Optional[Union[LocalConfig, ProdConfig, TestConfig]] = None,
    ):
        self.engine: Optional[Engine] = None
        self.session: AsyncSession = None
        if app is not None:
            self.init_app(app=app, config=config)

    def init_app(
        self, app: FastAPI, config: Union[LocalConfig, TestConfig, ProdConfig]
    ):
        print(">>> Current config status:", config)
        self.engine = create_engine(
            config.mysql_root_url.replace("aiomysql", "pymysql"),
            echo=config.db_echo,
            pool_recycle=config.db_pool_recycle,
            pool_pre_ping=True,
        )  # Root user
        while True:  # Connection check
            try:
                assert MySQL.is_db_exists(
                    self.engine, database_name=config.mysql_database
                ), f"Database {config.mysql_database} does not exists."
            except Exception as e:
                print(e)
                sleep(5)
            else:
                break

        if config.test_mode:  # Test mode
            assert isinstance(
                config, TestConfig
            ), "Config with 'test_mode == True' must be TestConfig! "
            assert (
                self.engine.url.host == "localhost"
            ), "DB host must be 'localhost' in test environment!"
            if not MySQL.is_db_exists(
                self.engine, database_name=config.mysql_test_database
            ):
                MySQL.create_db(self.engine, database_name=config.mysql_test_database)
            Base.metadata.drop_all(self.engine)
            Base.metadata.create_all(self.engine)
        else:  # Production or Local mode
            assert isinstance(
                config, Union[LocalConfig, ProdConfig]
            ), "Config with 'test_mode == False' must be LocalConfig or ProdConfig!"
            assert MySQL.is_db_exists(
                self.engine, database_name=config.mysql_database
            ), f"Database {config.mysql_database} does not exists!"
            assert self.engine.url.username != "root", "Database user must not be root!"
        self.engine.dispose()
        self.engine = create_async_engine(
            config.mysql_url if not config.test_mode else config.mysql_test_url,
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

        @app.on_event("startup")
        async def startup():
            # self.engine.connect()
            logging.critical(">>> DB connected")

        @app.on_event("shutdown")
        async def shutdown():
            self.session.remove()
            self.engine.dispose()
            logging.critical(">>> DB disconnected")

    async def get_db(self) -> AsyncSession:
        async with self.session() as session:
            yield session


db = SQLAlchemy()
