from time import time

from fastapi import HTTPException
from starlette.datastructures import Headers, QueryParams
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.common.config import EXCEPT_PATH_LIST, EXCEPT_PATH_REGEX
from app.database.crud.api_keys import get_api_key_and_owner
from app.errors.api_exceptions import (
    APIException,
    InternalServerError,
    Responses_400,
    Responses_401,
    exception_handler,
)
from app.models.base_models import UserToken
from app.utils.auth.token import token_decode
from app.utils.date_utils import UTC
from app.utils.logger import ApiLogger
from app.utils.params_utils import hash_params


class StateManager:
    @staticmethod
    def init(request: Request):
        request.state.req_time = UTC.now()
        request.state.start = time()
        if request.client is not None:
            # Check headers for real ip
            request.state.ip = request.headers.get(
                "x-forwarded-for", request.client.host
            ).split(",")[0]
        request.state.inspect = None
        request.state.user = None


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
    def non_api_service(
        headers: Headers,
        cookies: dict[str, str],
    ) -> UserToken:
        if "authorization" in headers.keys():
            token = headers["authorization"]
        elif "Authorization" in cookies.keys():
            token = cookies["Authorization"]
        else:
            raise Responses_401.not_authorized
        return Validator.jwt(token)


class Validator:
    @staticmethod
    async def api_key(
        api_access_key: str,
        hashed_secret: str,
        query_params: QueryParams | str,
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
        now_timestamp: int = UTC.timestamp()
        if not (now_timestamp - 60 < int(timestamp) < now_timestamp + 60):
            raise Responses_401.invalid_timestamp
        return UserToken(**matched_user.to_dict())

    @staticmethod
    def jwt(
        authorization: str,
    ) -> UserToken:
        token_info: dict = token_decode(authorization=authorization)
        return UserToken(**token_info)


async def access_control(request: Request, call_next: RequestResponseEndpoint):
    StateManager.init(request=request)
    url: str = request.url.path

    try:
        if EXCEPT_PATH_REGEX.match(url) is not None:
            ...  # Regex-whitelist endpoint
        elif url in EXCEPT_PATH_LIST:
            ...  # Whitelist endpoint
        elif url.startswith("/api/services"):
            # Api-service endpoint (required: accesskey + secretkey)
            request.state.user = await AccessControl.api_service(
                query_params=request.query_params,
                headers=request.headers,
            )
        else:
            # Non Api-service endpoint (required: jwttoken)
            request.state.user = AccessControl.non_api_service(
                headers=request.headers,
                cookies=request.cookies,
            )
        response: Response = await call_next(
            request
        )  # actual endpoint response

    except Exception as exception:  # If any error occurs...
        error: HTTPException | InternalServerError | APIException = (
            exception_handler(error=exception)
        )
        response: Response = JSONResponse(
            status_code=error.status_code,
            content={
                "status": error.status_code,
                "msg": error.msg
                if not isinstance(error, HTTPException)
                else None,
                "detail": error.detail
                if not isinstance(error, HTTPException)
                else error.detail,
                "code": error.code
                if not isinstance(error, HTTPException)
                else None,
            },
        )
        ApiLogger.clog(
            request=request,
            response=response,
            error=error,
            cookies=request.cookies,
            headers=dict(request.headers),
            query_params=dict(request.query_params),
        )
        return response  # The final response from server
    else:
        # Log error or service info
        if url.startswith("/api/services"):
            ApiLogger.clog(
                request=request,
                response=response,
                cookies=request.cookies,
                headers=dict(request.headers),
                query_params=dict(request.query_params),
            )
        return response  # The final response from server
