from tests.conftest import *
from asyncio import run
import pytest
from fastapi import status
from httpx import AsyncClient
from app.database.schema import Users, db
from app.common.config import ERROR_RESPONSES


@pytest.mark.asyncio
async def test_registration(client: AsyncClient, user_1):
    user = user_1
    response = await client.post("api/auth/register/email", json=user)
    response_body = response.json()
    assert response.status_code == status.HTTP_201_CREATED
    assert "Authorization" in response_body.keys()


def test_login(client, user_1):
    user = user_1
    response = client.post("api/auth/login/email", json=user)
    response_body = response.json()
    assert response.status_code == status.HTTP_200_OK
    assert "Authorization" in response_body.keys()


def test_registration_exist_email(client, user_2):
    new_user = user_2
    run(Users.create_new(db.get_db(), auto_commit=True, **new_user))
    response = client.post("api/auth/register/email", json=new_user)
    response_body = response.json()
    assert response.status_code == ERROR_RESPONSES["email_already_taken"]["status_code"]
    assert response_body["detail"] == ERROR_RESPONSES["email_already_taken"]["detail"]
