from typing import Optional, Tuple
from sqlalchemy import select, func, exists
from app.models.base_models import AddApiKey
from app.errors.api_exceptions import (
    Responses_400,
    Responses_404,
    Responses_500,
)
from app.common.config import MAX_API_KEY
from app.database.connection import db
from app.database.schemas.auth import ApiKeys, Users
from app.utils.auth.api_keys import generate_new_api_key


async def create_api_key(
    user_id: int,
    additional_key_info: AddApiKey,
) -> ApiKeys:
    if db.session is None:
        raise Responses_500.database_not_initialized
    async with db.session() as transaction:
        api_key_count_stmt = select(func.count(ApiKeys.id)).filter_by(user_id=user_id)
        api_key_count: int | None = await transaction.scalar(api_key_count_stmt)
        if api_key_count is not None and api_key_count >= MAX_API_KEY:
            raise Responses_400.max_key_count_exceed
        while True:
            new_api_key: ApiKeys = generate_new_api_key(
                user_id=user_id, additional_key_info=additional_key_info
            )
            is_api_key_duplicate_stmt = select(
                exists().where(ApiKeys.access_key == new_api_key.access_key)
            )
            is_api_key_duplicate: bool | None = await transaction.scalar(
                is_api_key_duplicate_stmt
            )
            if not is_api_key_duplicate:
                break
        transaction.add(new_api_key)
        await transaction.commit()
        await transaction.refresh(new_api_key)
        return new_api_key


async def get_api_keys(user_id: int) -> list[ApiKeys]:
    return await ApiKeys.fetchall_filtered_by(user_id=user_id)


async def get_api_key_owner(access_key: str) -> Users:
    if db.session is None:
        raise Responses_500.database_not_initialized
    async with db.session() as transaction:
        matched_api_key: Optional[ApiKeys] = await transaction.scalar(
            select(ApiKeys).filter_by(access_key=access_key)
        )
        if matched_api_key is None:
            raise Responses_404.not_found_access_key
        owner: Users = await Users.first_filtered_by(id=matched_api_key.user_id)
        if owner is None:
            raise Responses_404.not_found_user
        return owner


async def get_api_key_and_owner(access_key: str) -> Tuple[ApiKeys, Users]:
    if db.session is None:
        raise Responses_500.database_not_initialized
    async with db.session() as transaction:
        matched_api_key: Optional[ApiKeys] = await transaction.scalar(
            select(ApiKeys).filter_by(access_key=access_key)
        )
        if matched_api_key is None:
            raise Responses_404.not_found_access_key
        api_key_owner: Optional[Users] = await transaction.scalar(
            select(Users).filter_by(id=matched_api_key.user_id)
        )
        if api_key_owner is None:
            raise Responses_404.not_found_user
        return matched_api_key, api_key_owner


async def update_api_key(
    updated_key_info: dict,
    access_key_id: int,
    user_id: int,
) -> ApiKeys:
    if db.session is None:
        raise Responses_500.database_not_initialized
    async with db.session() as transaction:
        matched_api_key: Optional[ApiKeys] = await transaction.scalar(
            select(ApiKeys).filter_by(id=access_key_id, user_id=user_id)
        )
        if matched_api_key is None:
            raise Responses_404.not_found_api_key
        matched_api_key.set_values_as(**updated_key_info)
        transaction.add(matched_api_key)
        await transaction.commit()
        await transaction.refresh(matched_api_key)
        return matched_api_key


async def delete_api_key(
    access_key_id: int,
    access_key: str,
    user_id: int,
) -> None:
    if db.session is None:
        raise Responses_500.database_not_initialized
    async with db.session() as transaction:
        matched_api_key: Optional[ApiKeys] = await transaction.scalar(
            select(ApiKeys).filter_by(
                id=access_key_id, user_id=user_id, access_key=access_key
            )
        )
        if matched_api_key is None:
            raise Responses_404.not_found_api_key
        await transaction.delete(matched_api_key)
        await transaction.commit()
