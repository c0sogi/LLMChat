from httpx import AsyncClient
import pytest
from fastapi import status
from app.errors.api_exceptions import Responses_400
from app.utils.tests.random import random_user_generator


@pytest.mark.asyncio
async def test_register(async_client):
    valid_emails = ("ankitrai326@gmail.com", "my.ownsite@our-earth.org", "aa@a.a")
    invalid_emails = ("ankitrai326gmail.com", "ankitrai326.com")

    valid_users = [random_user_generator(email=email) for email in valid_emails]
    invalid_users = [random_user_generator(email=email) for email in invalid_emails]

    # Valid email
    for valid_user in valid_users:
        response = await async_client.post("api/auth/register/email", json=valid_user)
        response_body = response.json()
        assert response.status_code == status.HTTP_201_CREATED
        assert "Authorization" in response_body.keys(), response.content.decode("utf-8")

    # Invalid email
    for invalid_user in invalid_users:
        response = await async_client.post("api/auth/register/email", json=invalid_user)
        response_body = response.json()
        assert response.status_code != status.HTTP_201_CREATED
        assert "Authorization" not in response_body.keys(), response.content.decode("utf-8")


@pytest.mark.asyncio
async def test_auth(async_client: AsyncClient):
    new_user = random_user_generator()

    # Register
    response = await async_client.post("api/auth/register/email", json=new_user)
    assert response.status_code == status.HTTP_201_CREATED, response.content.decode("utf-8")

    # Login
    response = await async_client.post("api/auth/login/email", json=new_user)
    response_body = response.json()
    assert response.status_code == status.HTTP_200_OK, "User login faiture"
    assert "Authorization" in response_body.keys(), response.content.decode("utf-8")

    # Duplicate email check
    response = await async_client.post("api/auth/register/email", json=new_user)
    response_body = response.json()
    assert response.status_code == Responses_400.email_already_exists.status_code, "Email duplication check failure"
    assert response_body["detail"] == Responses_400.email_already_exists.detail, "Email duplication check failure"
