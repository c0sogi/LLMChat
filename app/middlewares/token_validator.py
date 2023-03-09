from time import time
from re import match
from typing import Union, Tuple, Optional
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from app.common.config import (
    config,
    EXCEPT_PATH_LIST,
    EXCEPT_PATH_REGEX,
)
from app.database.crud import get_api_key_and_owner
from app.errors import exceptions as ex
from app.errors.exceptions import (
    APIException,
    Responses_400,
    Responses_401,
    InternalServerError,
)
from app.models import UserToken
from app.utils.date_utils import UTC
from app.utils.logger import api_logger
from app.utils.query_utils import row_to_dict
from app.utils.encoding_and_hashing import hash_params, token_decode


async def access_control(request: Request, call_next: RequestResponseEndpoint):
    headers, cookies = request.headers, request.cookies
    url = request.url.path
    query_params = str(request.query_params)
    ip = request.client.host

    response: Optional[Response] = None
    error: Optional[Union[InternalServerError, APIException]] = None
    request.state.req_time = UTC.now()
    request.state.start = time()
    request.state.inspect = None
    request.state.user = None
    request.state.service = None
    request.state.ip = ip.split(",")[0] if "," in ip else ip

    try:
        if await url_pattern_check(url, EXCEPT_PATH_REGEX) or url in EXCEPT_PATH_LIST:
            response = await call_next(request)
            if url != "/":
                await api_logger(request=request, response=response)
            return response

        if url.startswith("/api/services") and not config.test_mode:
            access_key, timestamp = await queries_params_to_key_and_timestamp(
                query_params
            )
            if "secret" not in headers.keys():
                raise Responses_401.invalid_api_header
            request.state.user: UserToken = await validate_api_key(
                api_access_key=access_key,
                query_params=query_params,
                hashed_secret=headers["secret"],
                timestamp=timestamp,
            )

        else:
            # Validate token by headers and cookies
            if "authorization" in headers.keys():
                token = headers.get("authorization")
            elif "Authorization" in cookies.keys():
                token = cookies.get("Authorization")
            else:
                raise Responses_401.not_authorized
            request.state.user: UserToken = await validate_jwt(token)

        response = await call_next(request)

    except Exception as exception:  # If any error occurs...
        error: Union[
            Exception, InternalServerError, APIException
        ] = await exception_handler(error=exception)
        response = JSONResponse(
            status_code=error.status_code,
            content={
                "status": error.status_code,
                "msg": error.msg,
                "detail": error.detail,
                "code": error.code,
            },
        )
    finally:
        await api_logger(
            request=request,
            response=response,
            error=error,
            cookies=cookies,
            headers=dict(headers),
            query_params=query_params,
        ) if url.startswith("/api/services") or error is not None else ...
        return response


async def validate_api_key(
    api_access_key: str,
    hashed_secret: str,
    query_params: str,
    timestamp: str,
) -> UserToken:
    matched_api_key, matched_user = await get_api_key_and_owner(
        access_key=api_access_key
    )
    if not hashed_secret == hash_params(
        query_params=query_params, secret_key=matched_api_key.secret_key
    ):
        raise Responses_401.invalid_api_header
    now_timestamp: int = UTC.timestamp(hour_diff=9)
    if not (now_timestamp - 60 < int(timestamp) < now_timestamp + 60):
        raise Responses_401.invalid_timestamp
    return UserToken(**row_to_dict(matched_user))


async def validate_jwt(
    authorization: str,
) -> UserToken:
    token_info: dict = await token_decode(authorization=authorization)
    return UserToken(**token_info)


async def queries_params_to_key_and_timestamp(query_params: str) -> Tuple[str, str]:
    try:
        qs_dict = {
            qs_split.split("=")[0]: qs_split.split("=")[1]
            for qs_split in query_params.split("&")
        }
    except Exception:
        raise Responses_400.invalid_api_query
    if "key" not in qs_dict.keys() or "timestamp" not in qs_dict.keys():
        raise Responses_400.invalid_api_query
    return qs_dict["key"], qs_dict["timestamp"]


async def url_pattern_check(path: str, pattern: str) -> bool:
    return True if match(pattern, path) else False


async def exception_handler(
    error: Exception,
) -> Union[InternalServerError, APIException]:
    if isinstance(error, APIException):
        if error.status_code == 500:
            return InternalServerError(ex=error)
        else:
            return error
    elif isinstance(error, OperationalError):
        return InternalServerError(ex=error)
    elif isinstance(error, HTTPException):
        return error
    else:
        return InternalServerError()
