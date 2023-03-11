from __future__ import annotations
from dataclasses import dataclass, asdict, field
from os import environ
from typing import Union, Optional
from pathlib import Path


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=True)
class ErrorResponse:
    status_code: int
    detail: str

    @property
    def asdict(self) -> dict:
        return asdict(self)


DATABASE_URL_FORMAT: str = (
    "{dialect}+{driver}://{user}:{password}@{host}:3306/{database}?charset=utf8mb4"
)
MYSQL_ROOT_PASSWORD: str = environ.get("MYSQL_ROOT_PASSWORD")
MYSQL_USER: str = environ.get("MYSQL_USER")
MYSQL_PASSWORD: str = environ.get("MYSQL_PASSWORD")
MYSQL_DATABASE: str = environ.get("MYSQL_DATABASE")
MYSQL_TEST_DATABASE: str = environ.get("MYSQL_TEST_DATABASE")
MYSQL_HOST: str = "db"
DOMAIN: str = "walabi.store"
JWT_SECRET: str = environ.get("JWT_SECRET")
AWS_ACCESS_KEY: str = environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY: str = environ.get("AWS_SECRET_KEY")
AWS_AUTHORIZED_EMAIL: str = environ.get("AWS_AUTHORIZED_EMAIL")
SAMPLE_JWT_TOKEN: str = environ.get("SAMPLE_JWT_TOKEN")
SAMPLE_ACCESS_KEY: str = environ.get("SAMPLE_ACCESS_KEY")
SAMPLE_SECRET_KEY: str = environ.get("SAMPLE_SECRET_KEY")
KAKAO_RESTAPI_TOKEN: str = environ.get("KAKAO_RESTAPI_TOKEN")
JWT_ALGORITHM = "HS256"
EXCEPT_PATH_LIST = ["/", "/openapi.json"]
EXCEPT_PATH_REGEX = "^(/docs|/redoc|/api/auth|/favicon.ico)"
MAX_API_KEY = 3
MAX_API_WHITELIST = 10
KAKAO_IMAGE_URL = "http://k.kakaocdn.net/dn/wwWjr/btrYVhCnZDF/2bgXDJth2LyIajIjILhLK0/kakaolink40_original.png"
BASE_DIR = Path(__file__).parents[2]

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


ERROR_RESPONSES = {
    "no_email_or_password": ErrorResponse(400, "Email and PW must be provided.").asdict,
    "email_already_taken": ErrorResponse(400, "Email already exists.").asdict,
    "not_supported_feature": ErrorResponse(400, "Not supported.").asdict,
    "no_matched_user": ErrorResponse(400, "No matched user.").asdict,
    "enforce_domain_wildcard": ErrorResponse(
        500, "Domain wildcard patterns must be like '*.example.com'."
    ).asdict,
}


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
        option: Optional[str] = None,
    ) -> Union[LocalConfig, ProdConfig, TestConfig]:
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
            f"*.{DOMAIN}",
            DOMAIN,
            "localhost",
        ]
    )
    allowed_sites: list = field(
        default_factory=lambda: [
            f"*.{DOMAIN}",
            DOMAIN,
            "localhost",
        ]
    )


@dataclass(frozen=True)
class TestConfig(Config, metaclass=SingletonMetaClass):
    test_mode: bool = True
    mysql_host: str = "localhost"
    mysql_database: str = MYSQL_TEST_DATABASE


if __name__ == "__main__":
    print(Config())
