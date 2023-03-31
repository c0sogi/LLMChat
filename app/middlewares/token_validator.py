from datetime import datetime
from time import time
from types import FrameType
from typing import Optional
from fastapi import HTTPException
from starlette.datastructures import QueryParams, Headers
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from app.common.config import (
    EXCEPT_PATH_LIST,
    EXCEPT_PATH_REGEX,
)
from app.database.crud import get_api_key_and_owner
from app.errors.api_exceptions import (
    APIException,
    Responses_400,
    Responses_401,
    InternalServerError,
    exception_handler,
)
from app.models.base_models import UserToken
from app.utils.date_utils import UTC
from app.utils.logger import api_logger
from app.utils.query_utils import row_to_dict
from app.utils.encoding_and_hashing import hash_params, token_decode


class StateManager:
    @staticmethod
    async def init(request: Request):
        request.state.req_time: datetime = UTC.now()
        request.state.start: float = time()
        request.state.ip: str = (
            request.client.host.split(",")[0]
            if "," in request.client.host
            else request.client.host
        )
        request.state.inspect: Optional[FrameType] = None
        request.state.user: Optional[UserToken] = None


class AccessControl:
    @staticmethod
    async def api_service(
        query_params: QueryParams,
        headers: Headers,
    ) -> UserToken:
        query_params_dict: dict = dict(query_params)
        for query_key in ("key", "timestamp"):
            if query_key not in query_params_dict.keys():
                raise Responses_400.invalid_api_query
        for header_key in ("secret",):
            if header_key not in headers.keys():
                raise Responses_401.invalid_api_header
        return await Validator.api_key(
            query_params=query_params,
            api_access_key=query_params_dict["key"],
            timestamp=query_params_dict["timestamp"],
            hashed_secret=headers["secret"],
        )

    @staticmethod
    async def non_api_service(
        headers: Headers,
        cookies: dict[str, str],
    ) -> UserToken:
        if "authorization" in headers.keys():
            token = headers.get("authorization")
        elif "Authorization" in cookies.keys():
            token = cookies.get("Authorization")
        else:
            raise Responses_401.not_authorized
        return await Validator.jwt(token)


class Validator:
    @staticmethod
    async def api_key(
        api_access_key: str,
        hashed_secret: str,
        query_params: QueryParams,
        timestamp: str,
    ) -> UserToken:
        matched_api_key, matched_user = await get_api_key_and_owner(
            access_key=api_access_key
        )
        if not hashed_secret == hash_params(
            query_params=str(query_params),
            secret_key=matched_api_key.secret_key,
        ):
            raise Responses_401.invalid_api_header
        now_timestamp: int = UTC.timestamp(hour_diff=9)
        if not (now_timestamp - 60 < int(timestamp) < now_timestamp + 60):
            raise Responses_401.invalid_timestamp
        return UserToken(**row_to_dict(matched_user))

    @staticmethod
    async def jwt(
        authorization: str,
    ) -> UserToken:
        token_info: dict = await token_decode(authorization=authorization)
        return UserToken(**token_info)


async def access_control(request: Request, call_next: RequestResponseEndpoint):
    await StateManager.init(request=request)
    url: str = request.url.path
    error: Optional[InternalServerError | HTTPException | APIException] = None
    response: Optional[Response] = None

    try:

        if EXCEPT_PATH_REGEX.match(url) is not None:
            ...  # Regex-whitelist endpoint
        elif url in EXCEPT_PATH_LIST:
            ...  # Whitelist endpoint
        elif url.startswith("/api/services"):
            # Api-service endpoint (required: accesskey + secretkey)
            request.state.user: UserToken = await AccessControl.api_service(
                query_params=request.query_params,
                headers=request.headers,
            )
        else:
            # Non Api-service endpoint (required: jwttoken)
            request.state.user: UserToken = await AccessControl.non_api_service(
                headers=request.headers,
                cookies=request.cookies,
            )
        response = await call_next(request)  # actual endpoint response

    except Exception as exception:  # If any error occurs...
        error: HTTPException | InternalServerError | APIException = (
            await exception_handler(error=exception)
        )
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
        # Log error or service info
        if url.startswith("/api/services") or error is not None:
            await api_logger(
                request=request,
                response=response,
                error=error,
                cookies=request.cookies,
                headers=dict(request.headers),
                query_params=dict(request.query_params),
            )
        return response  # The final response from server
