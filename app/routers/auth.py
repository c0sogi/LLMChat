import bcrypt
from fastapi import APIRouter, Response
from fastapi.requests import Request
from app.common.config import TOKEN_EXPIRE_HOURS
from app.errors.error_responses import ErrorResponses
from app.database.crud import is_email_exist, register_new_user
from app.database.schema import Users
from app.models import SnsType, Token, UserRegister, UserToken
from app.utils.encoding_and_hashing import create_access_token

router = APIRouter(prefix="/auth")

"""
1. 구글 로그인을 위한 구글 앱 준비 (구글 개발자 도구)
2. FB 로그인을 위한 FB 앱 준비 (FB 개발자 도구)
3. 카카오 로그인을 위한 카카오 앱준비( 카카오 개발자 도구)
4. 이메일, 비밀번호로 가입 (v)
5. 가입된 이메일, 비밀번호로 로그인, (v)
6. JWT 발급 (v)

7. 이메일 인증 실패시 이메일 변경
8. 이메일 인증 메일 발송
9. 각 SNS 에서 Unlink
10. 회원 탈퇴
11. 탈퇴 회원 정보 저장 기간 동안 보유(법적 최대 한도 내에서, 가입 때 약관 동의 받아야 함, 재가입 방지 용도로 사용하면 가능)
"""


@router.post("/register/{sns_type}", status_code=201, response_model=Token)
async def register(
    request: Request,
    response: Response,
    sns_type: SnsType,
    reg_info: UserRegister,
) -> Token:
    if sns_type == SnsType.email:
        if not (reg_info.email and reg_info.password):
            raise ErrorResponses.no_email_or_password
        if await is_email_exist(reg_info.email):
            raise ErrorResponses.email_already_taken
        hashed_password: bytes = bcrypt.hashpw(
            password=reg_info.password.encode("utf-8"),
            salt=bcrypt.gensalt(),
        )
        new_user: Users = await register_new_user(
            email=reg_info.email,
            hashed_password=hashed_password,
            ip_address=request.client.host,
        )
        data_to_be_tokenized: dict = UserToken.from_orm(new_user).dict(
            exclude={"password", "marketing_agree"}
        )
        token: str = create_access_token(
            data=data_to_be_tokenized, expires_delta=TOKEN_EXPIRE_HOURS
        )
        response.set_cookie(
            key="Authorization",
            value=f"Bearer {token}",
            max_age=TOKEN_EXPIRE_HOURS * 3600,
            secure=True,
            httponly=True,
        )
        return Token(Authorization=f"Bearer {token}")
    raise ErrorResponses.not_supported_feature


@router.post("/login/{sns_type}", status_code=200, response_model=Token)
async def login(
    response: Response,
    sns_type: SnsType,
    user_info: UserRegister,
) -> Token:
    if sns_type == SnsType.email:
        if not (user_info.email and user_info.password):
            raise ErrorResponses.no_email_or_password
        matched_user: Users = await Users.first_filtered_by(email=user_info.email)
        if matched_user is None:
            raise ErrorResponses.no_matched_user
        if not bcrypt.checkpw(
            password=user_info.password.encode("utf-8"),
            hashed_password=matched_user.password.encode("utf-8"),
        ):
            raise ErrorResponses.no_matched_user
        data_to_be_tokenized: dict = UserToken.from_orm(matched_user).dict(
            exclude={"password", "marketing_agree"}
        )
        token: str = create_access_token(
            data=data_to_be_tokenized, expires_delta=TOKEN_EXPIRE_HOURS
        )
        response.set_cookie(
            key="Authorization",
            value=f"Bearer {token}",
            max_age=TOKEN_EXPIRE_HOURS * 3600,
            secure=True,
            httponly=True,
        )
        return Token(Authorization=f"Bearer {token}")
    else:
        raise ErrorResponses.not_supported_feature
