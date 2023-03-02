from app.common.config import MAX_API_KEY, MAX_API_WHITELIST


class StatusCode:
    HTTP_500 = 500
    HTTP_400 = 400
    HTTP_401 = 401
    HTTP_403 = 403
    HTTP_404 = 404
    HTTP_405 = 405


class APIException(Exception):
    status_code: int
    code: str
    msg: str
    detail: str
    ex: Exception

    def __init__(
        self,
        *,
        status_code: int = StatusCode.HTTP_500,
        code: str = "000000",
        msg: str = None,
        detail: str = None,
        ex: Exception = None,
    ):
        self.status_code = status_code
        self.code = code
        self.msg = msg
        self.detail = detail
        self.ex = ex
        super().__init__(ex)


class NotFoundUserEx(APIException):
    def __init__(self, user_id: int = None, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_404,
            msg="해당 유저를 찾을 수 없습니다.",
            detail="Not Found User ID : {user_id}",
            code="{StatusCode.HTTP_400}{'1'.zfill(4)}",
            ex=ex,
        )


class NotAuthorized(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_401,
            msg="로그인이 필요한 서비스 입니다.",
            detail="Authorization Required",
            code="{StatusCode.HTTP_401}{'1'.zfill(4)}",
            ex=ex,
        )


class TokenExpiredEx(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_400,
            msg="세션이 만료되어 로그아웃 되었습니다.",
            detail="Token Expired",
            code="{StatusCode.HTTP_400}{'1'.zfill(4)}",
            ex=ex,
        )


class TokenDecodeEx(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_400,
            msg="비정상적인 접근입니다.",
            detail="Token has been compromised.",
            code="{StatusCode.HTTP_400}{'2'.zfill(4)}",
            ex=ex,
        )


class NoKeyMatchEx(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_404,
            msg="해당 키에 대한 권한이 없거나 해당 키가 없습니다.",
            detail="No Keys Matched",
            code="{StatusCode.HTTP_404}{'3'.zfill(4)}",
            ex=ex,
        )


class MaxKeyCountEx(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_400,
            msg=f"API 키 생성은 {MAX_API_KEY}개 까지 가능합니다.",
            detail="Max Key Count Reached",
            code="{StatusCode.HTTP_400}{'4'.zfill(4)}",
            ex=ex,
        )


class MaxWLCountEx(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_400,
            msg=f"화이트리스트 생성은 {MAX_API_WHITELIST}개 까지 가능합니다.",
            detail="Max Whitelist Count Reached",
            code="{StatusCode.HTTP_400}{'5'.zfill(4)}",
            ex=ex,
        )


class InvalidIpEx(APIException):
    def __init__(self, ip: str, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_400,
            msg="{ip}는 올바른 IP 가 아닙니다.",
            detail="invalid IP : {ip}",
            code="{StatusCode.HTTP_400}{'6'.zfill(4)}",
            ex=ex,
        )


class SqlFailureEx(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_500,
            msg="이 에러는 서버측 에러 입니다. 자동으로 리포팅 되며, 빠르게 수정하겠습니다.",
            detail="Internal Server Error",
            code="{StatusCode.HTTP_500}{'2'.zfill(4)}",
            ex=ex,
        )


class APIQueryStringEx(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_400,
            msg="쿼리스트링은 key, timestamp 2개만 허용되며, 2개 모두 요청시 제출되어야 합니다.",
            detail="Query String Only Accept key and timestamp.",
            code="{StatusCode.HTTP_400}{'7'.zfill(4)}",
            ex=ex,
        )


class APIHeaderInvalidEx(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_400,
            msg="헤더에 키 해싱된 Secret 이 없거나, 유효하지 않습니다.",
            detail="Invalid HMAC secret in Header",
            code="{StatusCode.HTTP_400}{'8'.zfill(4)}",
            ex=ex,
        )


class APITimestampEx(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_400,
            msg="쿼리스트링에 포함된 타임스탬프는 KST 이며, 현재 시간보다 작아야 하고, 현재시간 - 10초 보다는 커야 합니다.",
            detail="timestamp in Query String must be KST, Timestamp must be less than now, and greater than now - 10.",
            code="{StatusCode.HTTP_400}{'9'.zfill(4)}",
            ex=ex,
        )


class NotFoundAccessKeyEx(APIException):
    def __init__(self, api_key: str, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_404,
            msg="API 키를 찾을 수 없습니다.",
            detail="Not found such API Access Key : {api_key}",
            code="{StatusCode.HTTP_404}{'10'.zfill(4)}",
            ex=ex,
        )


class KakaoSendFailureEx(APIException):
    def __init__(self, ex: Exception = None):
        super().__init__(
            status_code=StatusCode.HTTP_400,
            msg="카카오톡 전송에 실패했습니다.",
            detail="Failed to send KAKAO MSG.",
            code="{StatusCode.HTTP_400}{'11'.zfill(4)}",
            ex=ex,
        )
