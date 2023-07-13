import bcrypt
from fastapi import APIRouter, Response, Security, status
from fastapi.requests import Request

from app.common.config import TOKEN_EXPIRE_HOURS
from app.database.crud.users import is_email_exist, register_new_user
from app.database.schemas.auth import Users
from app.dependencies import USER_DEPENDENCY
from app.errors.api_exceptions import Responses_400, Responses_404
from app.models.base_models import SnsType, Token, UserRegister, UserToken
from app.utils.auth.register_validation import (
    is_email_length_in_range,
    is_email_valid_format,
    is_password_length_in_range,
)
from app.utils.auth.token import create_access_token, token_decode
from app.utils.chat.managers.cache import CacheManager

router = APIRouter(prefix="/auth")


@router.post("/register/{sns_type}", status_code=201, response_model=Token)
async def register(
    request: Request,
    response: Response,
    sns_type: SnsType,
    reg_info: UserRegister,
) -> Token:
    if sns_type == SnsType.EMAIL:
        if not (reg_info.email and reg_info.password):
            raise Responses_400.no_email_or_password

        if is_email_length_in_range(email=reg_info.email) is False:
            raise Responses_400.email_length_not_in_range

        if is_password_length_in_range(password=reg_info.password) is False:
            raise Responses_400.password_length_not_in_range

        if is_email_valid_format(email=reg_info.email) is False:
            raise Responses_400.invalid_email_format

        if await is_email_exist(email=reg_info.email):
            raise Responses_400.email_already_exists

        hashed_password: str = bcrypt.hashpw(
            password=reg_info.password.encode("utf-8"),
            salt=bcrypt.gensalt(),
        ).decode("utf-8")
        new_user: Users = await register_new_user(
            email=reg_info.email,
            hashed_password=hashed_password,
            ip_address=request.state.ip,
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
    raise Responses_400.not_supported_feature


@router.delete("/register", status_code=status.HTTP_204_NO_CONTENT)
async def unregister(
    authorization: str = Security(USER_DEPENDENCY),
):
    registered_user: UserToken = UserToken(**token_decode(authorization))
    if registered_user.email is not None:
        await CacheManager.delete_user(user_id=registered_user.email)
    await Users.delete_filtered(
        Users.email == registered_user.email, autocommit=True
    )


@router.post("/login/{sns_type}", status_code=200, response_model=Token)
async def login(
    response: Response,
    sns_type: SnsType,
    user_info: UserRegister,
) -> Token:
    if sns_type == SnsType.EMAIL:
        if not (user_info.email and user_info.password):
            raise Responses_400.no_email_or_password
        matched_user: Users = await Users.first_filtered_by(
            email=user_info.email
        )
        if matched_user is None or matched_user.password is None:
            raise Responses_404.not_found_user
        if not bcrypt.checkpw(
            password=user_info.password.encode("utf-8"),
            hashed_password=matched_user.password.encode("utf-8"),
        ):
            raise Responses_404.not_found_user
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
        raise Responses_400.not_supported_feature
