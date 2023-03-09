from typing import Optional, Tuple, List
from sqlalchemy import select, func, exists
from app.models import AddApiKey
from app.errors.exceptions import (
    Responses_400,
    Responses_404,
)
from app.common.config import MAX_API_KEY, MAX_API_WHITELIST
from app.database.schema import db, Users, ApiKeys, ApiWhiteLists
from app.utils.encoding_and_hashing import generate_api_key

# from sqlalchemy.ext.asyncio import AsyncSession


async def is_email_exist(email: str) -> bool:
    return (
        True if await db.scalar(select(exists().where(Users.email == email))) else False
    )


async def get_me(user_id: int):
    return await Users.first_filtered_by(id=user_id)


async def get_api_key_and_owner(access_key: str) -> Tuple[ApiKeys, Users]:
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


async def register_new_user(email: str, hashed_password: str, ip_address: str) -> Users:
    return await Users.add_one(
        autocommit=True,
        refresh=True,
        email=email,
        password=hashed_password,
        ip_address=ip_address,
    )


async def find_matched_user(email: str) -> Users:
    return await Users.first_filtered_by(email=email)


async def create_api_key(user_id: int, additional_key_info: AddApiKey) -> ApiKeys:
    async with db.session() as transaction:
        api_key_count_stmt = select(func.count(ApiKeys.id)).filter_by(user_id=user_id)
        api_key_count: int = await transaction.scalar(api_key_count_stmt)
        if api_key_count >= MAX_API_KEY:
            raise Responses_400.max_key_count_exceed
        while True:
            new_api_key: ApiKeys = await generate_api_key(
                user_id=user_id, additional_key_info=additional_key_info
            )
            is_api_key_duplicate_stmt = select(
                exists().where(ApiKeys.access_key == new_api_key.access_key)
            )
            is_api_key_duplicate: bool = await transaction.scalar(
                is_api_key_duplicate_stmt
            )
            if not is_api_key_duplicate:
                break
        transaction.add(new_api_key)
        await transaction.commit()
        await transaction.refresh(new_api_key)
        return new_api_key


async def get_api_keys(user_id: int) -> List[ApiKeys]:
    return await ApiKeys.fetchall_filtered_by(user_id=user_id)


async def update_api_key(
    updated_key_info: dict, access_key_id: int, user_id: int
) -> ApiKeys:
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


async def delete_api_key(access_key_id: int, access_key: str, user_id: int) -> None:
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


async def create_api_key_whitelist(ip_address: str, api_key_id: int) -> ApiWhiteLists:
    async with db.session() as transaction:
        whitelist_count_stmt = select(func.count(ApiWhiteLists.id)).filter_by(
            api_key_id=api_key_id
        )
        whitelist_count: int = await transaction.scalar(whitelist_count_stmt)
        if whitelist_count >= MAX_API_WHITELIST:
            raise Responses_400.max_whitekey_count_exceed
        ip_duplicated_whitelist_stmt = select(ApiWhiteLists).filter_by(
            api_key_id=api_key_id, ip_address=ip_address
        )
        ip_duplicated_whitelist = await transaction.scalar(ip_duplicated_whitelist_stmt)
        if ip_duplicated_whitelist is not None:
            return ip_duplicated_whitelist
        new_whitelist = ApiWhiteLists(api_key_id=api_key_id, ip_address=ip_address)
        transaction.add(new_whitelist)
        await transaction.commit()
        await transaction.refresh(new_whitelist)
        return new_whitelist


async def get_api_key_whitelist(api_key_id: int) -> List[ApiKeys]:
    return await ApiWhiteLists.fetchall_filtered_by(api_key_id=api_key_id)


async def delete_api_key_whitelist(
    user_id: int, api_key_id: int, whitelist_id: int
) -> None:
    async with db.session() as transaction:
        matched_api_key: Optional[ApiKeys] = await transaction.scalar(
            select(ApiKeys).filter_by(id=api_key_id, user_id=user_id)
        )
        if matched_api_key is None:
            raise Responses_404.not_found_api_key
        matched_whitelist_stmt = select(ApiWhiteLists).filter_by(
            id=whitelist_id, api_key_id=api_key_id
        )
        matched_whitelist = await transaction.scalar(matched_whitelist_stmt)
        await transaction.delete(matched_whitelist)
        await transaction.commit()
