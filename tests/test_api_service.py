from typing import Literal
from httpx import AsyncClient, Response
import pytest
from datetime import datetime
from app.utils.date_utils import UTC
from app.utils.encoding_and_hashing import hash_params
from app.utils.query_utils import parse_params
import logging


async def get_apikey(client: AsyncClient, authorized_header: str):
    api_key_memo: str = f"TESTING : {str(datetime.now())}"
    response: Response = await client.post(
        "/api/user/apikeys",
        json={"user_memo": api_key_memo},
        headers=authorized_header,
    )
    response_body = response.json()
    assert response.status_code == 200
    assert "access_key" in response_body
    assert "secret_key" in response_body
    apikey = {
        "access_key": response_body["access_key"],
        "secret_key": response_body["secret_key"],
    }

    response = await client.get("/api/user/apikeys", headers=authorized_header)
    response_body = response.json()
    assert response.status_code == 200
    assert api_key_memo in response_body[0]["user_memo"]
    return apikey


async def request_service(
    client: AsyncClient,
    access_key: str,
    secret_key: str,
    request_method: Literal["get", "post", "put", "delete", "options"],
    allowed_status_codes: tuple[int] = (200, 307),
    service_name: str = "",
    required_query_params: dict[str, any] = {},
    required_headers: dict[str, any] = {},
    stream: bool = False,
) -> any:
    all_query_params: str = parse_params(
        params={
            "key": access_key,
            "timestamp": UTC.timestamp(hour_diff=9),
        }
        | required_query_params
    )
    method_options: dict = {
        "headers": {
            "secret": hash_params(
                query_params=all_query_params,
                secret_key=secret_key,
            )
        }
        | required_headers
    }
    url: str = f"/api/services/{service_name}?{all_query_params}"

    if stream:
        response_body = ""
        async with client.stream(
            method=request_method.upper(), url=url, **method_options
        ) as response:
            assert response.status_code in allowed_status_codes
            async for chunk in response.aiter_text():
                response_body += chunk
                print("Streamed data:", chunk)
    else:
        response: Response = await getattr(client, request_method.lower())(
            url=url, **method_options
        )
        response_body: any = response.json()
        logging.info(msg=response_body)
        assert response.status_code in allowed_status_codes
    return response_body


@pytest.mark.asyncio
async def test_request_api(login_header, client):
    api_key_dict: dict[str, any] = await get_apikey(
        client=client, authorized_header=login_header
    )
    access_key, secret_key = api_key_dict["access_key"], api_key_dict["secret_key"]
    service_name: str = ""
    request_method: str = "get"
    required_query_params: dict[str, any] = {}
    required_headers: dict[str, any] = {}
    allowed_status_codes: tuple[int] = (200, 307)
    await request_service(
        client=client,
        access_key=access_key,
        secret_key=secret_key,
        request_method=request_method,
        allowed_status_codes=allowed_status_codes,
        service_name=service_name,
        required_query_params=required_query_params,
        required_headers=required_headers,
        stream=False,
    )


@pytest.mark.asyncio
async def test_weather_api(login_header: dict[str, str], client: AsyncClient):
    api_key_dict: dict[str, any] = await get_apikey(
        client=client, authorized_header=login_header
    )
    access_key, secret_key = api_key_dict["access_key"], api_key_dict["secret_key"]
    service_name: str = "weather"
    request_method: str = "get"
    required_query_params: dict[str, any] = {
        "latitude": 37.0,
        "longitude": 120.0,
    }
    required_headers: dict[str, any] = {}
    allowed_status_codes: tuple[int] = (200, 307)
    await request_service(
        client=client,
        access_key=access_key,
        secret_key=secret_key,
        request_method=request_method,
        allowed_status_codes=allowed_status_codes,
        service_name=service_name,
        required_query_params=required_query_params,
        required_headers=required_headers,
        stream=False,
    )


@pytest.mark.asyncio
async def test_stream_api(login_header: dict[str, str], client: AsyncClient):
    api_key_dict: dict[str, any] = await get_apikey(
        client=client, authorized_header=login_header
    )
    access_key, secret_key = api_key_dict["access_key"], api_key_dict["secret_key"]
    service_name: str = "stream"
    request_method: str = "get"
    required_query_params: dict[str, any] = {}
    required_headers: dict[str, any] = {}
    allowed_status_codes: tuple[int] = (200, 307)
    await request_service(
        client=client,
        access_key=access_key,
        secret_key=secret_key,
        request_method=request_method,
        allowed_status_codes=allowed_status_codes,
        service_name=service_name,
        required_query_params=required_query_params,
        required_headers=required_headers,
        stream=True,
    )
