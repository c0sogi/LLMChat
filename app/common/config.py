from __future__ import annotations
from dataclasses import dataclass, asdict  # field
from os import environ
from typing import Union, Optional
from pathlib import Path
from urllib import parse


# from app.utils.encoding_and_hashing import SecretConfigSetup
# password_from_environ = environ.get("SECRET_CONFIGS_PASSWORD", None)
# secret_config_setup = SecretConfigSetup(
#     password=password_from_environ
#     if password_from_environ is not None
#     else input("Enter Passwords:"),
#     json_file_name="secret_configs.json",
# )
#
#
# @dataclass(frozen=True)
# class SecretConfig(metaclass=SingletonMetaClass):
#     secret_config: dict = field(default_factory=secret_config_setup.initialize)


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


MYSQL_URL_FORMAT: str = "mysql+pymysql://{}:{}@{}:3306/{}?charset=utf8mb4"
MYSQL_ROOT_PASSWORD: str = environ.get("MYSQL_ROOT_PASSWORD")
MYSQL_USER: str = environ.get("MYSQL_USER")
MYSQL_PASSWORD: str = environ.get("MYSQL_PASSWORD")
MYSQL_DATABASE: str = environ.get("MYSQL_DATABASE")
MYSQL_TEST_DATABASE: str = environ.get("MYSQL_TEST_DATABASE")
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
    "no_email_or_pw": ErrorResponse(400, "Email and PW must be provided.").asdict,
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
    db_echo: bool = False
    debug: bool = False
    test_mode: bool = False
    mysql_database: str = MYSQL_DATABASE
    mysql_root_url: str = MYSQL_URL_FORMAT.format(
        "root", parse.quote(MYSQL_ROOT_PASSWORD), "db", "mysql"
    )

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
    mysql_url: str = MYSQL_URL_FORMAT.format(
        MYSQL_USER, parse.quote(MYSQL_PASSWORD), "db", MYSQL_DATABASE
    )
    trusted_hosts = ["*"]
    allowed_sites = ["*"]


@dataclass(frozen=True)
class ProdConfig(Config, metaclass=SingletonMetaClass):
    mysql_url: str = MYSQL_URL_FORMAT.format(
        MYSQL_USER, parse.quote(MYSQL_PASSWORD), "db", MYSQL_DATABASE
    )
    trusted_hosts = [
        "*.walabi.store",
        "walabi.store",
        "localhost",
        "127.0.0.1",
        "192.168.0.3",
    ]
    allowed_sites = [
        "*.walabi.store",
        "walabi.store",
        "localhost",
        "127.0.0.1",
        "192.168.0.3",
    ]


@dataclass(frozen=True)
class TestConfig(Config, metaclass=SingletonMetaClass):
    test_mode: bool = True
    mysql_root_url: str = MYSQL_URL_FORMAT.format(
        "root", parse.quote(MYSQL_ROOT_PASSWORD), "localhost", "mysql"
    )
    mysql_test_url: str = MYSQL_URL_FORMAT.format(
        "root", parse.quote(MYSQL_ROOT_PASSWORD), "localhost", MYSQL_TEST_DATABASE
    )
    mysql_test_database: str = MYSQL_TEST_DATABASE
    trusted_hosts = ["*"]
    allowed_sites = ["*"]
