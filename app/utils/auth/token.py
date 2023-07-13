from datetime import datetime, timedelta

from jwt import decode as jwt_decode
from jwt import encode as jwt_encode
from jwt.exceptions import DecodeError, ExpiredSignatureError

from app.common.config import JWT_ALGORITHM, JWT_SECRET
from app.errors.api_exceptions import Responses_401


def create_access_token(
    *, data: dict, expires_delta: int | None = None
) -> str:
    to_encode: dict = data.copy()
    if expires_delta is not None and expires_delta != 0:
        to_encode.update(
            {"exp": datetime.utcnow() + timedelta(hours=expires_delta)}
        )
    return jwt_encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def token_decode(authorization: str) -> dict:
    if authorization is None:
        raise Responses_401.token_decode_failure
    try:
        authorization = authorization.replace("Bearer ", "")
        payload = jwt_decode(
            authorization, key=JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
    except ExpiredSignatureError:
        raise Responses_401.token_expired
    except DecodeError:
        raise Responses_401.token_decode_failure
    return payload
