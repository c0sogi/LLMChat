import asyncio
from datetime import datetime
from typing import AsyncGenerator, Generator
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import httpx
import pytest_asyncio
from app.utils.auth.token import create_access_token
from app.utils.tests.random import random_user_generator
from app.database.schemas.auth import Users
from app.common.app_settings import create_app
from app.common.config import Config, LoggingConfig
from app.utils.logger import CustomLogger
from app.models.base_models import UserToken
from app.utils.chat.managers.cache import CacheManager, cache


@pytest.fixture(scope="session")
def config():
    return Config.get(option="test")


@pytest.fixture(scope="session")
def cache_manager():
    cache.start(config=Config.get(option="test"))
    return CacheManager


@pytest.fixture(scope="session")
def test_logger():
    return CustomLogger(
        name="PyTest", logging_config=LoggingConfig(file_log_name="./logs/test.log")
    )


@pytest.fixture(scope="session", autouse=True)
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def app(config) -> FastAPI:
    return create_app(config)


@pytest.fixture(scope="session")
def base_http_url() -> str:
    return "http://localhost"


@pytest.fixture(scope="session")
def base_websocket_url() -> str:
    return "ws://localhost"


@pytest_asyncio.fixture(scope="session")
async def async_client(
    app: FastAPI, base_http_url: str
) -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(app=app, base_url=base_http_url) as ac:
        yield ac


@pytest.fixture(scope="session")
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(app=app) as tc:
        yield tc


@pytest_asyncio.fixture(scope="session")
async def login_header(random_user: dict[str, str]) -> dict[str, str]:
    """
    테스트 전 사용자 미리 등록
    """
    new_user = await Users.add_one(autocommit=True, refresh=True, **random_user)  # type: ignore
    access_token = create_access_token(
        data=UserToken.from_orm(new_user).dict(exclude={"password", "marketing_agree"}),
        expires_delta=24,
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest_asyncio.fixture(scope="session")
async def api_key_dict(async_client: httpx.AsyncClient, login_header: dict[str, str]):
    api_key_memo: str = f"TESTING : {str(datetime.now())}"
    response: httpx.Response = await async_client.post(
        "/api/user/apikeys",
        json={"user_memo": api_key_memo},
        headers=login_header,
    )
    response_body = response.json()
    assert response.status_code == 201
    assert "access_key" in response_body
    assert "secret_key" in response_body
    apikey = {
        "access_key": response_body["access_key"],
        "secret_key": response_body["secret_key"],
    }

    response = await async_client.get("/api/user/apikeys", headers=login_header)
    response_body = response.json()
    assert response.status_code == 200
    assert api_key_memo in response_body[0]["user_memo"]
    return apikey


@pytest.fixture(scope="session")
def random_user():
    return random_user_generator()


def pytest_collection_modifyitems(items):
    app_tests = []
    redis_tests = []
    other_tests = []
    for item in items:
        if "client" in item.fixturenames:
            app_tests.append(item)
        elif item.get_closest_marker("redistest") is not None:
            redis_tests.append(item)
        else:
            other_tests.append(item)
    items[:] = redis_tests + app_tests + other_tests
