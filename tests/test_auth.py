import pytest
from fastapi import status

from app.errors.exceptions import Responses_400


# @pytest.mark.asyncio
# async def test_registration(client, random_user):
#     user = random_user
#     response = await client.post("api/auth/register/email", json=user)
#     response_body = response.json()
#     assert response.status_code == status.HTTP_201_CREATED
#     assert "Authorization" in response_body.keys()


@pytest.mark.asyncio
async def test_auth(client, random_user):
    user = random_user
    response = await client.post("api/auth/register/email", json=user)
    assert response.status_code == status.HTTP_201_CREATED, "User register failure"
    response = await client.post("api/auth/login/email", json=user)
    response_body = response.json()
    assert response.status_code == status.HTTP_200_OK, "User login faiture"
    assert "Authorization" in response_body.keys(), "No JWT token issued"
    response = await client.post("api/auth/register/email", json=user)
    response_body = response.json()
    assert (
        response.status_code == Responses_400.email_already_exists.status_code
    ), "Email duplication check failure"
    assert (
        response_body["detail"] == Responses_400.email_already_exists.detail
    ), "Email duplication check failure"
