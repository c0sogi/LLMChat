from typing import Any, Literal
from httpx import AsyncClient, Response
import pytest
from app.utils.date_utils import UTC
from app.utils.params_utils import hash_params, parse_params


async def request_service(
    async_client: AsyncClient,
    access_key: str,
    secret_key: str,
    request_method: Literal["get", "post", "put", "delete", "options"],
    allowed_status_codes: tuple = (200, 201, 307),
    service_name: str = "",
    required_query_params: dict[str, Any] = {},
    required_headers: dict[str, Any] = {},
    stream: bool = False,
    logger=None,
) -> Any:
    all_query_params: str = parse_params(
        params={
            "key": access_key,
            "timestamp": UTC.timestamp(),
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
        async with async_client.stream(method=request_method.upper(), url=url, **method_options) as response:
            assert response.status_code in allowed_status_codes
            async for chunk in response.aiter_text():
                response_body += chunk
                logger.info(f"Streamed data: {chunk}") if logger is not None else ...
    else:
        response: Response = await getattr(async_client, request_method.lower())(url=url, **method_options)
        response_body: Any = response.json()
        logger.info(f"response_body: {response_body}") if logger is not None else ...
        assert response.status_code in allowed_status_codes
    return response_body


@pytest.mark.asyncio
async def test_request_api(async_client: AsyncClient, api_key_dict: dict, test_logger):
    access_key, secret_key = api_key_dict["access_key"], api_key_dict["secret_key"]
    service_name: str = ""
    request_method: str = "get"
    required_query_params: dict[str, Any] = {}
    required_headers: dict[str, Any] = {}
    allowed_status_codes: tuple = (200, 201, 307)
    await request_service(
        async_client=async_client,
        access_key=access_key,
        secret_key=secret_key,
        request_method=request_method,
        allowed_status_codes=allowed_status_codes,
        service_name=service_name,
        required_query_params=required_query_params,
        required_headers=required_headers,
        stream=False,
        logger=test_logger,
    )


# @pytest.mark.asyncio
# async def test_weather_api(async_client: AsyncClient, api_key_dict: dict):
#     access_key, secret_key = api_key_dict["access_key"], api_key_dict["secret_key"]
#     service_name: str = "weather"
#     request_method: str = "get"
#     required_query_params: dict[str, any] = {
#         "latitude": 37.0,
#         "longitude": 120.0,
#     }
#     required_headers: dict[str, any] = {}
#     allowed_status_codes: tuple[int] = (200, 307)
#     await request_service(
#         async_client=async_client,
#         access_key=access_key,
#         secret_key=secret_key,
#         request_method=request_method,
#         allowed_status_codes=allowed_status_codes,
#         service_name=service_name,
#         required_query_params=required_query_params,
#         required_headers=required_headers,
#         stream=False,
#     )


# @pytest.mark.asyncio
# async def test_stream_api(async_client: AsyncClient, api_key_dict: dict):
#     access_key, secret_key = api_key_dict["access_key"], api_key_dict["secret_key"]
#     service_name: str = "stream"
#     request_method: str = "get"
#     required_query_params: dict[str, any] = {}
#     required_headers: dict[str, any] = {}
#     allowed_status_codes: tuple[int] = (200, 307)
#     await request_service(
#         async_client=async_client,
#         access_key=access_key,
#         secret_key=secret_key,
#         request_method=request_method,
#         allowed_status_codes=allowed_status_codes,
#         service_name=service_name,
#         required_query_params=required_query_params,
#         required_headers=required_headers,
#         stream=True,
#     )
