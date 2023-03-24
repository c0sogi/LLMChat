from __future__ import annotations
from dataclasses import dataclass, field
from os import environ
from pathlib import Path


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


DATABASE_URL_FORMAT: str = (
    "{dialect}+{driver}://{user}:{password}@{host}:3306/{database}?charset=utf8mb4"
)
MYSQL_ROOT_PASSWORD: str = environ.get("MYSQL_ROOT_PASSWORD")
MYSQL_USER: str = environ.get("MYSQL_USER")
MYSQL_PASSWORD: str = environ.get("MYSQL_PASSWORD")
MYSQL_DATABASE: str = environ.get("MYSQL_DATABASE")
MYSQL_TEST_DATABASE: str = environ.get("MYSQL_TEST_DATABASE")
MYSQL_HOST: str = "db"
HOST_MAIN: str = environ.get("HOST_MAIN")
JWT_SECRET: str = environ.get("JWT_SECRET")
AWS_ACCESS_KEY: str = environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY: str = environ.get("AWS_SECRET_KEY")
AWS_AUTHORIZED_EMAIL: str = environ.get("AWS_AUTHORIZED_EMAIL")
SAMPLE_JWT_TOKEN: str = environ.get("SAMPLE_JWT_TOKEN")
SAMPLE_ACCESS_KEY: str = environ.get("SAMPLE_ACCESS_KEY")
SAMPLE_SECRET_KEY: str = environ.get("SAMPLE_SECRET_KEY")
KAKAO_RESTAPI_TOKEN: str = environ.get("KAKAO_RESTAPI_TOKEN")
WEATHERBIT_API_KEY: str = environ.get("WEATHERBIT_API_KEY")
JWT_ALGORITHM: str = "HS256"
EXCEPT_PATH_LIST: list = ["/", "/openapi.json", "/test"]
EXCEPT_PATH_REGEX: str = "^(/docs|/redoc|/api/auth|/favicon.ico|/chatgpt)"
TOKEN_EXPIRE_HOURS: int = 2
MAX_API_KEY: int = 3
MAX_API_WHITELIST: int = 10
KAKAO_IMAGE_URL: str = "http://k.kakaocdn.net/dn/wwWjr/btrYVhCnZDF/2bgXDJth2LyIajIjILhLK0/kakaolink40_original.png"
BASE_DIR: str = Path(__file__).parents[2]

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
    db_pool_recycle: int = 900
    db_echo: bool = True
    debug: bool = False
    test_mode: bool = False
    database_url_format: str = DATABASE_URL_FORMAT
    mysql_root_password: str = MYSQL_ROOT_PASSWORD
    mysql_user: str = MYSQL_USER
    mysql_password: str = MYSQL_PASSWORD
    mysql_database: str = MYSQL_DATABASE
    mysql_host: str = MYSQL_HOST
    trusted_hosts: list = field(default_factory=lambda: ["*"])
    allowed_sites: list = field(default_factory=lambda: ["*"])

    @staticmethod
    def get(
        option: str | None = None,
    ) -> LocalConfig | ProdConfig | TestConfig:
        config_key = option if option is not None else environ.get("API_ENV", "local")
        return {"prod": ProdConfig, "local": LocalConfig, "test": TestConfig}[
            config_key
        ]()


@dataclass(frozen=True)
class LocalConfig(Config, metaclass=SingletonMetaClass):
    debug: bool = True
    mysql_host: str = "localhost"


@dataclass(frozen=True)
class ProdConfig(Config, metaclass=SingletonMetaClass):
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


config = Config.get()
