from __future__ import annotations

import logging
from dataclasses import dataclass, field
from os import environ
from pathlib import Path
from re import Pattern, compile


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# API Server Variables

EXCEPT_PATH_LIST: tuple = (
    "/",
    "/openapi.json",
    "/test",
)
EXCEPT_PATH_REGEX: Pattern = compile("^(/docs|/redoc|/api/auth|/favicon.ico|/chatgpt|/flutter_service_worker.js)")
TOKEN_EXPIRE_HOURS: int = 168
MAX_API_KEY: int = 3
MAX_API_WHITELIST: int = 10
BASE_DIR: Path = Path(__file__).parents[2]

# MySQL Variables
MYSQL_URL_FORMAT: str = "{dialect}+{driver}://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
MYSQL_ROOT_PASSWORD: str = environ["MYSQL_ROOT_PASSWORD"]
MYSQL_USER: str = environ["MYSQL_USER"]
MYSQL_PASSWORD: str = environ.get("MYSQL_PASSWORD", "")
MYSQL_DATABASE: str = environ["MYSQL_DATABASE"]
MYSQL_TEST_DATABASE: str = environ.get("MYSQL_TEST_DATABASE", MYSQL_DATABASE)
MYSQL_HOST: str = "db"
MYSQL_PORT: int = int(environ.get("MYSQL_PORT", 3306))

# Redis Variables
REDIS_HOST: str = "cache"
REDIS_PORT: int = int(environ.get("REDIS_PORT", 6379))
REDIS_DATABASE: int = int(environ.get("REDIS_DATABASE", 0))
REDIS_PASSWORD: str = environ["REDIS_PASSWORD"]

# Other Required Variables
HOST_MAIN: str = environ.get("HOST_MAIN", "localhost:8000")
JWT_SECRET: str = environ["JWT_SECRET"]
JWT_ALGORITHM: str = "HS256"


# Optional Service Variables
OPENAI_API_KEY: str | None = environ.get("OPENAI_API_KEY")
GOOGLE_TRANSLATE_API_KEY: str | None = environ.get("GOOGLE_TRANSLATE_API_KEY")
PAPAGO_CLIENT_ID: str | None = environ.get("PAPAGO_CLIENT_ID")
PAPAGO_CLIENT_SECRET: str | None = environ.get("PAPAGO_CLIENT_SECRET")
AWS_ACCESS_KEY: str | None = environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY: str | None = environ.get("AWS_SECRET_KEY")
AWS_AUTHORIZED_EMAIL: str | None = environ.get("AWS_AUTHORIZED_EMAIL")
SAMPLE_JWT_TOKEN: str | None = environ.get("SAMPLE_JWT_TOKEN")
SAMPLE_ACCESS_KEY: str | None = environ.get("SAMPLE_ACCESS_KEY")
SAMPLE_SECRET_KEY: str | None = environ.get("SAMPLE_SECRET_KEY")
KAKAO_RESTAPI_TOKEN: str | None = environ.get("KAKAO_RESTAPI_TOKEN")
WEATHERBIT_API_KEY: str | None = environ.get("WEATHERBIT_API_KEY")
KAKAO_IMAGE_URL: str | None = (
    "http://k.kakaocdn.net/dn/wwWjr/btrYVhCnZDF/2bgXDJth2LyIajIjILhLK0/kakaolink40_original.png"
)

"""
400 Bad Request
401 Unauthorized
403 Forbidden
404 Not Found
405 Method not allowed
500 Internal Error
502 Bad Gateway
504 Timeout
200 OK
201 Created
"""


@dataclass(frozen=True)
class Config(metaclass=SingletonMetaClass):
    app_title: str = f"{HOST_MAIN} API service"
    app_description: str = ""
    app_version: str = "0.9.0"
    db_pool_recycle: int = 900
    db_echo: bool = False
    debug: bool = False
    test_mode: bool = False
    database_url_format: str = MYSQL_URL_FORMAT
    mysql_root_password: str = MYSQL_ROOT_PASSWORD
    mysql_user: str = MYSQL_USER
    mysql_password: str = MYSQL_PASSWORD
    mysql_database: str = MYSQL_DATABASE
    mysql_host: str = MYSQL_HOST
    mysql_port: int = MYSQL_PORT
    redis_host: str = REDIS_HOST
    redis_port: int = REDIS_PORT
    redis_database: int = REDIS_DATABASE
    redis_password: str = REDIS_PASSWORD
    trusted_hosts: list[str] = field(default_factory=lambda: ["*"])
    allowed_sites: list[str] = field(default_factory=lambda: ["*"])

    @staticmethod
    def get(
        option: str | None = None,
    ) -> LocalConfig | ProdConfig | TestConfig:
        config_key = option if option is not None else environ.get("API_ENV", "local")
        config_key = "test" if environ.get("PYTEST_RUNNING") is not None else config_key
        return {"prod": ProdConfig, "local": LocalConfig, "test": TestConfig}[config_key]()


@dataclass(frozen=True)
class LocalConfig(Config, metaclass=SingletonMetaClass):
    debug: bool = True


@dataclass(frozen=True)
class ProdConfig(Config, metaclass=SingletonMetaClass):
    db_echo: bool = False
    trusted_hosts: list = field(
        default_factory=lambda: [
            f"*.{HOST_MAIN}",
            HOST_MAIN,
            "localhost",
        ]
    )
    allowed_sites: list = field(
        default_factory=lambda: [
            f"*.{HOST_MAIN}",
            HOST_MAIN,
            "localhost",
        ]
    )


@dataclass(frozen=True)
class TestConfig(Config, metaclass=SingletonMetaClass):
    test_mode: bool = True
    debug: bool = False
    mysql_host: str = "localhost"
    mysql_database: str = MYSQL_TEST_DATABASE
    redis_host: str = "localhost"


@dataclass(frozen=True)
class LoggingConfig:
    logger_level: int = logging.DEBUG
    console_log_level: int = logging.INFO
    file_log_level: int | None = logging.DEBUG
    file_log_name: str | None = "./logs/debug.log"
    logging_format: str = "[%(asctime)s] %(name)s:%(levelname)s - %(message)s"


config = Config.get()
logging_config = LoggingConfig()

if __name__ == "__main__":
    print(BASE_DIR / "app")
