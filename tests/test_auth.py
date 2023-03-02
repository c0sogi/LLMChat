from tests.test_configuration import *
from fastapi import status
from asyncio import run
from app.database.schema import Users
from app.common.config import ERROR_RESPONSES


def test_registration(client, user_1):
    user = user_1
    response = client.post("api/auth/register/email", json=user)
    response_body = response.json()
    assert response.status_code == status.HTTP_201_CREATED
    assert "Authorization" in response_body.keys()


def test_login(client, user_1):
    user = user_1
    response = client.post("api/auth/login/email", json=user)
    response_body = response.json()
    assert response.status_code == status.HTTP_200_OK
    assert "Authorization" in response_body.keys()


def test_registration_exist_email(client, session, user_2):
    new_user = user_2
    run(Users.create_schema_instance(session, auto_commit=True, **new_user))
    response = client.post("api/auth/register/email", json=new_user)
    response_body = response.json()
    assert response.status_code == ERROR_RESPONSES["email_already_taken"]["status_code"]
    assert response_body["detail"] == ERROR_RESPONSES["email_already_taken"]["detail"]
