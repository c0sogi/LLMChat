from datetime import datetime, timedelta
import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.requests import Request
from app.common.config import ERROR_RESPONSES, JWT_ALGORITHM, JWT_SECRET
from app.database.crud import is_email_exist, register_new_user
from app.database.schema import db, Users, AsyncSession
from app.models import SnsType, Token, UserRegister, UserToken
from app.utils.logger import logger

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


@router.post(
    "/register/{sns_type}", status_code=status.HTTP_201_CREATED, response_model=Token
)
async def register(
    sns_type: SnsType,
    reg_info: UserRegister,
    request: Request,
) -> dict:
    if sns_type == SnsType.email:
        if not (reg_info.email and reg_info.password):
            raise HTTPException(**ERROR_RESPONSES["no_email_or_password"])
        if await is_email_exist(reg_info.email):
            raise HTTPException(**ERROR_RESPONSES["email_already_taken"])
        hashed_password = bcrypt.hashpw(
            reg_info.password.encode("utf-8"), bcrypt.gensalt()
        )
        new_user = await register_new_user(
            email=reg_info.email,
            hashed_password=hashed_password,
            ip_address=request.client.host,
        )
        data_to_be_tokenized = UserToken.from_orm(new_user).dict(
            exclude={"password", "marketing_agree"}
        )
        return {
            "Authorization": f"Bearer {create_access_token(data=data_to_be_tokenized)}"
        }  # token
    raise HTTPException(**ERROR_RESPONSES["not_supported_feature"])


@router.post("/login/{sns_type}", status_code=200, response_model=Token)
async def login(
    sns_type: SnsType,
    user_info: UserRegister,
) -> dict:
    if sns_type == SnsType.email:
        if not (user_info.email and user_info.password):
            raise HTTPException(**ERROR_RESPONSES["no_email_or_password"])
        matched_user: Users = await Users.first_filtered_by(email=user_info.email)
        if matched_user is None:
            raise HTTPException(**ERROR_RESPONSES["no_matched_user"])
        if not bcrypt.checkpw(
            user_info.password.encode("utf-8"), matched_user.password.encode("utf-8")
        ):
            raise HTTPException(**ERROR_RESPONSES["no_matched_user"])
        data_to_be_tokenized = UserToken.from_orm(matched_user).dict(
            exclude={"password", "marketing_agree"}
        )
        return {
            "Authorization": f"Bearer {create_access_token(data=data_to_be_tokenized)}"
        }  # token
    else:
        raise HTTPException(**ERROR_RESPONSES["not_supported_feature"])


# from sqlalchemy.engine import ChunkedIteratorResult
def create_access_token(*, data: dict = None, expires_delta: int = None) -> str:
    to_encode = data.copy()
    if expires_delta is not None:
        to_encode.update({"exp": datetime.utcnow() + timedelta(hours=expires_delta)})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
