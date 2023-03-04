import pytest
from asyncio import run
from fastapi.testclient import TestClient
from app.database.schema import Users, db
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
    return create_app(Config.get(option="test"))


@pytest.fixture(scope="session")
def client(app):
    return TestClient(app=app)


@pytest.fixture(scope="function")
def session():
    if db.session_local is None:
        raise Exception("must be called 'init_app'")
    local_session = db.session_local()
    try:
        yield local_session
    finally:
        local_session.close()


@pytest.fixture(scope="function")
def login_header(session, user_1):
    """
    테스트 전 사용자 미리 등록
    """
    new_user = run(Users.create_schema_instance(session, auto_commit=True, **user_1))
    access_token = create_access_token(
        data=UserToken.from_orm(new_user).dict(exclude={"pw", "marketing_agree"}),
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture(scope="session")
def user_1():
    return {
        "email": "aaaa@test.com",
        "pw": "123",
        "name": "테스트 유저1",
        "phone": "01011111111",
    }


@pytest.fixture(scope="session")
def user_2():
    return {
        "email": "bbbb@test.com",
        "pw": "123",
        "name": "테스트 유저2",
        "phone": "01022222222",
    }
