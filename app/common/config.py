from dataclasses import dataclass, asdict  # field
from os import environ
from typing import Union, Optional
from pathlib import Path
from urllib import parse

# from app.utils.encoding_and_hashing import SecretConfigSetup


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


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
#

MYSQL_ROOT_PASSWORD: str = environ.get("MYSQL_ROOT_PASSWORD")
DB_URL: str = environ.get("DB_URL").format(parse.quote(MYSQL_ROOT_PASSWORD))
TEST_DB_URL: str = environ.get("TEST_DB_URL").format(parse.quote(MYSQL_ROOT_PASSWORD))
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
class ErrorResponse:
    status_code: int
    detail: str

    @property
    def asdict(self) -> dict:
        return asdict(self)


ERROR_RESPONSES = {
    "no_email_or_pw": ErrorResponse(400, "Email and PW must be provided.").asdict,
    "email_already_taken": ErrorResponse(400, "Email already exists.").asdict,
    "not_supported_feature": ErrorResponse(400, "Not supported.").asdict,
    "no_matched_user": ErrorResponse(400, "No matched user.").asdict,
    "enforce_domain_wildcard": ErrorResponse(
        500, "Domain wildcard patterns must be like '*.example.com'."
    ).asdict,
}

JWT_ALGORITHM = "HS256"
EXCEPT_PATH_LIST = ["/", "/openapi.json"]
EXCEPT_PATH_REGEX = "^(/docs|/redoc|/api/auth|/favicon.ico)"
MAX_API_KEY = 3
MAX_API_WHITELIST = 10
KAKAO_IMAGE_URL = "http://k.kakaocdn.net/dn/wwWjr/btrYVhCnZDF/2bgXDJth2LyIajIjILhLK0/kakaolink40_original.png"
BASE_DIR = Path(__file__).parents[2]


@dataclass(frozen=True)
class Config(metaclass=SingletonMetaClass):
    db_pool_recycle: int = 900
    db_echo: bool = False
    debug: bool = False
    test_mode: bool = False
    db_url: str = DB_URL
    schema_name: str = MYSQL_DATABASE

    @property
    def get(self):
        return {
            "prod": ProdConfig,
            "local": LocalConfig,
            "test": TestConfig,
        }[environ.get("API_ENV", "local")]()


@dataclass(frozen=True)
class LocalConfig(Config, metaclass=SingletonMetaClass):
    debug: bool = True
    trusted_hosts = ["*"]
    allowed_sites = ["*"]


@dataclass(frozen=True)
class ProdConfig(Config, metaclass=SingletonMetaClass):
    trusted_hosts = [
        "*.walabi.store",
        "walabi.store",
        "localhost",
        "127.0.0.1",
        "192.168.0.3",
        "152.67.192.154",
    ]
    allowed_sites = [
        "*.walabi.store",
        "walabi.store",
        "localhost",
        "127.0.0.1",
        "192.168.0.3",
        "152.67.192.154",
    ]


@dataclass(frozen=True)
class TestConfig(Config, metaclass=SingletonMetaClass):
    test_mode: bool = True
    test_schema_name: str = MYSQL_TEST_DATABASE
    test_db_url: str = TEST_DB_URL
    trusted_hosts = ["*"]
    allowed_sites = ["*"]


def get_config(
    option: Optional[str] = None,
) -> Union[LocalConfig, ProdConfig, TestConfig]:
    config_key = option if option is not None else environ.get("API_ENV", "local")
    return {"prod": ProdConfig, "local": LocalConfig, "test": TestConfig}[config_key]()
