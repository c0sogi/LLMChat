from dataclasses import dataclass
from typing import Optional

from fastapi.exceptions import HTTPException
from sqlalchemy.exc import OperationalError

from app.common.config import MAX_API_KEY, MAX_API_WHITELIST


def error_codes(status_code: int, internal_code: int) -> str:
    return f"{status_code}{str(internal_code).zfill(4)}"


class APIException(Exception):
    status_code: int = 500
    internal_code: int = 0
    msg: Optional[str]
    detail: Optional[str]
    ex: Optional[Exception]

    def __init__(
        self,
        *,
        status_code: int,
        internal_code: int,
        msg: Optional[str] = None,
        detail: Optional[str] = None,
        ex: Optional[Exception] = None,
    ):
        self.status_code = status_code
        self.code = error_codes(
            status_code=status_code, internal_code=internal_code
        )
        self.msg = msg
        self.detail = detail
        self.ex = ex
        super().__init__(ex)

    def __call__(
        self,
        lazy_format: Optional[dict[str, str]] = None,
        ex: Optional[Exception] = None,
    ) -> "APIException":
        if (
            self.msg is not None
            and self.detail is not None
            and lazy_format is not None
        ):  # lazy format for msg and detail
            self.msg = self.msg.format(**lazy_format)
            self.detail = self.detail.format(**lazy_format)
        if ex is not None:  # set exception if exists
            self.ex = ex
        return self


class InternalServerError(APIException):
    status_code: int = 500
    internal_code: int = 9999
    msg: str = "이 에러는 서버측 에러 입니다. 자동으로 리포팅 되며, 빠르게 수정하겠습니다."
    detail: str = "Internal Server Error"

    def __init__(self, ex: Optional[Exception] = None):
        super().__init__(
            status_code=self.status_code,
            internal_code=self.internal_code,
            msg=self.msg,
            detail=self.detail,
            ex=ex,
        )


class InvalidIpError(APIException):
    status_code: int = 400
    internal_code: int = 10
    msg: str = "{ip}는 올바른 IP 가 아닙니다."
    detail: str = "invalid IP : {ip}"

    def __init__(self, ip: str):
        super().__init__(
            status_code=400,
            internal_code=self.internal_code,
            msg=self.msg.format(ip=ip),
            detail=self.detail.format(ip=ip),
        )


@dataclass(frozen=True)
class Responses_400:
    """
    클라이언트가 잘못된 방식의 요청을 하고 있음
    """

    no_email_or_password: APIException = APIException(
        status_code=400,
        internal_code=1,
        msg="이메일과 비밀번호를 모두 입력해주세요.",
        detail="Email and PW must be provided.",
    )
    email_already_exists: APIException = APIException(
        status_code=400,
        internal_code=2,
        msg="해당 이메일은 이미 가입되어 있습니다.",
        detail="Email already exists.",
    )
    not_supported_feature: APIException = APIException(
        status_code=400,
        internal_code=3,
        msg="해당 기능은 아직 사용할 수 없습니다.",
        detail="Not supported feature.",
    )
    unregister_failure: APIException = APIException(
        status_code=400,
        internal_code=4,
        msg="회원 탈퇴에 실패했습니다.",
        detail="Failed to unregister.",
    )
    max_key_count_exceed: APIException = APIException(
        status_code=400,
        internal_code=8,
        msg=f"API 키 생성은 {MAX_API_KEY}개 까지 가능합니다.",
        detail="Max Key Count Reached",
    )
    max_whitekey_count_exceed: APIException = APIException(
        status_code=400,
        internal_code=9,
        msg=f"화이트리스트 생성은 {MAX_API_WHITELIST}개 까지 가능합니다.",
        detail="Max Whitelist Count Reached",
    )
    invalid_ip: APIException = APIException(
        status_code=400,
        internal_code=10,
        msg="{ip}는 올바른 IP 가 아닙니다.",
        detail="invalid IP : {ip}",
    )
    invalid_api_query: APIException = APIException(
        status_code=400,
        internal_code=11,
        msg="쿼리스트링은 key, timestamp 2개만 허용되며, 2개 모두 요청시 제출되어야 합니다.",
        detail="Query String Only Accept key and timestamp.",
    )
    kakao_send_failure: APIException = APIException(
        status_code=400,
        internal_code=15,
        msg="카카오톡 전송에 실패했습니다.",
        detail="Failed to send KAKAO MSG.",
    )
    websocket_in_use: APIException = APIException(
        status_code=400,
        internal_code=16,
        msg="이미 사용중인 웹소켓입니다.",
        detail="Websocket is already in use.",
    )
    invalid_email_format: APIException = APIException(
        status_code=400,
        internal_code=17,
        msg="올바르지 않은 이메일 형식 입니다.",
        detail="Invalid Email Format.",
    )
    email_length_not_in_range: APIException = APIException(
        status_code=400,
        internal_code=18,
        msg="이메일은 6자 이상 50자 이하로 입력해주세요.",
        detail="Email must be 6 ~ 50 characters.",
    )
    password_length_not_in_range: APIException = APIException(
        status_code=400,
        internal_code=19,
        msg="비밀번호는 6자 이상 100자 이하로 입력해주세요.",
        detail="Password must be 6 ~ 100 characters.",
    )


@dataclass(frozen=True)
class Responses_401:
    """
    클라이언트가 올바른 자격 증명을 제시해야 함 (보안 관련)
    """

    not_authorized: APIException = APIException(
        status_code=401,
        internal_code=1,
        msg="로그인이 필요한 서비스 입니다.",
        detail="Authorization Required",
    )
    token_expired: APIException = APIException(
        status_code=401,
        internal_code=6,
        msg="세션이 만료되어 로그아웃 되었습니다.",
        detail="Token Expired",
    )
    token_decode_failure: APIException = APIException(
        status_code=401,
        internal_code=7,
        msg="비정상적인 접근입니다.",
        detail="Token has been compromised.",
    )
    invalid_api_header: APIException = APIException(
        status_code=401,
        internal_code=12,
        msg="헤더에 키 해싱된 Secret 이 없거나, 유효하지 않습니다.",
        detail="Invalid HMAC secret in Header",
    )
    invalid_timestamp: APIException = APIException(
        status_code=401,
        internal_code=13,
        msg="쿼리스트링에 포함된 타임스탬프는 KST 이며, 현재 시간보다 작아야 하고, 현재시간 - 10초 보다는 커야 합니다.",
        detail="timestamp in Query String must be KST, Timestamp must be less than now, and greater than now - 10.",
    )


@dataclass(frozen=True)
class Responses_404:
    """
    클라이언트가 무언가를 요청하였으나 해당 항목에 대해 어떤 것도 찾지 못하였음
    """

    not_found_user: APIException = APIException(
        status_code=404,
        internal_code=5,
        msg="해당 유저를 찾을 수 없습니다.",
        detail="Not found user.",
    )
    not_found_access_key: APIException = APIException(
        status_code=404,
        internal_code=14,
        msg="해당 Access key과 일치하는 API 키를 찾을 수 없습니다.",
        detail="Not found such API Access Key",
    )
    not_found_api_key: APIException = APIException(
        status_code=404,
        internal_code=7,
        msg="제공된 조건에 부합하는 Api key를 찾을 수 없습니다.",
        detail="No API Key matched such conditions",
    )
    not_found_preset: APIException = APIException(
        status_code=404,
        internal_code=13,
        msg="제공된 조건에 부합하는 프리셋을 찾을 수 없습니다.",
        detail="No preset matched such conditions",
    )


@dataclass(frozen=True)
class Responses_500:
    """
    서버 내부적으로 오류가 발생하였음
    """

    middleware_exception: APIException = APIException(
        status_code=500,
        internal_code=2,
        detail="Middleware could not be initialized",
    )
    websocket_error: APIException = APIException(
        status_code=500,
        internal_code=3,
        msg="웹소켓 연결에 문제 발생",
        detail="Websocket error",
    )
    database_not_initialized: APIException = APIException(
        status_code=500,
        internal_code=4,
        msg="데이터베이스가 초기화 되지 않았습니다.",
        detail="Database not initialized",
    )
    cache_not_initialized: APIException = APIException(
        status_code=500,
        internal_code=5,
        msg="캐시가 초기화 되지 않았습니다.",
        detail="Cache not initialized",
    )
    vectorestore_not_initialized: APIException = APIException(
        status_code=500,
        internal_code=5,
        msg="벡터 저장소가 초기화 되지 않았습니다.",
        detail="Vector Store not initialized",
    )


def exception_handler(
    error: Exception,
) -> InternalServerError | HTTPException | APIException:
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
