from datetime import datetime, timedelta
import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.requests import Request
from app.common.config import ERROR_RESPONSES, JWT_ALGORITHM, JWT_SECRET
from app.database.connection import Session, db
from app.database.schema import Users
from app.models import SnsType, Token, UserRegister, UserToken

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
    session: Session = Depends(db.get_db),
) -> dict:
    if sns_type == SnsType.email:
        if (not reg_info.email) or (not reg_info.pw):
            raise HTTPException(**ERROR_RESPONSES["no_email_or_pw"])
        if await is_email_exist(reg_info.email):
            raise HTTPException(**ERROR_RESPONSES["email_already_taken"])
        hashed_pw = bcrypt.hashpw(reg_info.pw.encode("utf-8"), bcrypt.gensalt())
        new_user_db = await Users.create_schema_instance(
            session,
            auto_commit=True,
            pw=hashed_pw,
            email=reg_info.email,
            ip_address=request.client.host,
        )
        data_to_be_tokenized = UserToken.from_orm(new_user_db).dict(
            exclude={"pw", "marketing_agree"}
        )
        return {
            "Authorization": f"Bearer {create_access_token(data=data_to_be_tokenized)}"
        }  # token
    raise HTTPException(**ERROR_RESPONSES["not_supported_feature"])


@router.post("/login/{sns_type}", status_code=200, response_model=Token)
async def login(
    sns_type: SnsType,
    user_info: UserRegister,
    request: Request,
    session: Session = Depends(db.get_db),
) -> dict:
    if sns_type == SnsType.email:
        if (not user_info.email) or (not user_info.pw):
            raise HTTPException(**ERROR_RESPONSES["no_email_or_pw"])
        user_from_db = await Users.get_filtered_schema_instances(
            session, email=user_info.email
        )
        if user_from_db.count() == 0:
            raise HTTPException(**ERROR_RESPONSES["no_matched_user"])
        if not bcrypt.checkpw(
            user_info.pw.encode("utf-8"), user_from_db.first().pw.encode("utf-8")
        ):
            raise HTTPException(**ERROR_RESPONSES["no_matched_user"])
        data_to_be_tokenized = UserToken.from_orm(user_from_db.first()).dict(
            exclude={"pw", "marketing_agree"}
        )
        user_from_db.update(auto_commit=True, ip_address=request.client.host)
        return {
            "Authorization": f"Bearer {create_access_token(data=data_to_be_tokenized)}"
        }  # token
    else:
        raise HTTPException(**ERROR_RESPONSES["not_supported_feature"])


async def is_email_exist(email: str) -> bool:
    return True if (await Users.get_row_from_db(email=email)) else False


def create_access_token(*, data: dict = None, expires_delta: int = None) -> str:
    to_encode = data.copy()
    if expires_delta is not None:
        to_encode.update({"exp": datetime.utcnow() + timedelta(hours=expires_delta)})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
