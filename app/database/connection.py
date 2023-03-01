from fastapi import FastAPI
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import (sessionmaker, scoped_session, Session, declarative_base)
import logging
Base = declarative_base()


class MySQL:
    @staticmethod
    def is_db_exists(engine: Engine, schema_name: str) -> bool:
        query = text(f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{schema_name}';")
        with engine.connect() as conn:
            result_proxy = conn.execute(query)
            result = result_proxy.scalar()
            return bool(result)

    @staticmethod
    def drop_db(engine: Engine, schema_name: str) -> None:
        with engine.connect() as conn:
            conn.execute(text(f"DROP DATABASE {schema_name};"))

    @staticmethod
    def create_db(engine: Engine, schema_name: str) -> None:
        with engine.connect() as conn:
            print('>>> Creating database:', schema_name)
            conn.execute(text(f"CREATE DATABASE {schema_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;"))


class SQLAlchemy:
    def __init__(self, app: FastAPI = None, **kwargs):
        self.engine = None
        self.session_local = None
        if app is not None:
            self.init_app(app=app, **kwargs)

    def init_app(self, app: FastAPI, **kwargs):
        db_url = kwargs.get("db_url")
        pool_recycle = kwargs.setdefault("db_pool_recycle", 900)
        test_mode = kwargs.setdefault("test_mode", False)
        echo = kwargs.setdefault("db_echo", True)
        self.engine = create_engine(db_url, echo=echo, pool_recycle=pool_recycle, pool_pre_ping=True)

        if test_mode:  # create schema
            test_schema_name = kwargs.get('test_schema_name')
            test_db_url = kwargs.get('test_db_url')
            assert self.engine.url.host == "localhost", "db host must be 'localhost' in test environment"
            if MySQL.is_db_exists(self.engine, test_schema_name):
                MySQL.drop_db(self.engine, test_schema_name)
                MySQL.create_db(self.engine, test_schema_name)
            else:
                MySQL.create_db(self.engine, test_schema_name)
            self.engine.dispose()
            self.engine = create_engine(test_db_url, echo=echo, pool_recycle=pool_recycle, pool_pre_ping=True)
        Base().metadata.create_all(self.engine)
        self.session_local = scoped_session(sessionmaker(bind=self.engine, autocommit=False, autoflush=False))

        @app.on_event("startup")
        async def startup():
            self.engine.connect()
            logging.info("DB connected.")
            # create_task(background_task_state.run())

        @app.on_event("shutdown")
        async def shutdown():
            self.session_local.remove()
            self.engine.dispose()
            logging.info("DB disconnected")

    def get_db(self) -> Session:
        if self.session_local is None:
            raise Exception("must be called 'init_app'")
        local_session = self.session_local()
        try:
            yield local_session
        finally:
            local_session.close()


db = SQLAlchemy()
