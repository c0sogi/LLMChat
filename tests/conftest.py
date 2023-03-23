from os import environ

environ["API_ENV"] = "test"
from typing import AsyncGenerator
import pytest
from httpx import AsyncClient
import pytest_asyncio
from uuid import uuid4
from os import environ
from app.database.schema import Users
from app.common.app_settings import create_app
from app.common.config import config
from app.models import UserToken
from app.routers.auth import create_access_token


"""
1. DB 생성
2. 테이블 생성
3. 테스트 코드 작동
4. 테이블 레코드 삭제
"""


@pytest.fixture(scope="session")
def app():
    _app = create_app(config)
    return _app


@pytest_asyncio.fixture(scope="function")
async def client(app) -> AsyncGenerator:
    async with AsyncClient(app=app, base_url="http://localhost") as ac:
        yield ac


@pytest_asyncio.fixture(scope="function")
async def login_header(random_user: dict[str, str]) -> dict[str, str]:
    """
    테스트 전 사용자 미리 등록
    """
    new_user = await Users.add_one(autocommit=True, refresh=True, **random_user)
    access_token = create_access_token(
        data=UserToken.from_orm(new_user).dict(exclude={"password", "marketing_agree"}),
        expires_delta=24,
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture(scope="function")
def random_user():
    random_8_digits = str(hash(uuid4()))[:8]
    return {
        "email": f"{random_8_digits}@test.com",
        "password": "123",
        "name": f"{random_8_digits}",
        "phone_number": f"010{random_8_digits}",
    }
