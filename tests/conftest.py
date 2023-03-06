from typing import AsyncGenerator
import pytest
from httpx import AsyncClient
import pytest_asyncio
from uuid import uuid4
from app.database.schema import Users
from app.common.app_settings import create_app
from app.common.config import Config
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
    _app = create_app(Config.get(option="test"))
    return _app


# @pytest_asyncio.fixture(scope="function")
# async def session(self) -> AsyncSession:
#     async with self.session() as session:
#         yield session


@pytest_asyncio.fixture(scope="function")
async def client(app) -> AsyncGenerator:
    async with AsyncClient(app=app, base_url="http://localhost") as ac:
        yield ac


@pytest_asyncio.fixture(scope="function")
async def login_header(random_user):
    """
    테스트 전 사용자 미리 등록
    """
    new_user = await Users.create_new(auto_commit=True, **random_user)
    access_token = create_access_token(
        data=UserToken.from_orm(new_user).dict(exclude={"pw", "marketing_agree"}),
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture(scope="function")
def random_user():
    random_8_digits = str(hash(uuid4()))[:8]
    return {
        "email": f"{random_8_digits}@test.com",
        "pw": "123",
        "name": f"{random_8_digits}",
        "phone": f"010{random_8_digits}",
    }
