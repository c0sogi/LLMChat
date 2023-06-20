from typing import Optional
from sqlalchemy import select, func
from app.errors.api_exceptions import (
    Responses_400,
    Responses_404,
    Responses_500,
)
from app.common.config import MAX_API_WHITELIST
from app.database.connection import db
from app.database.schemas.auth import (
    ApiKeys,
    ApiWhiteLists,
)


async def create_api_key_whitelist(
    ip_address: str,
    api_key_id: int,
) -> ApiWhiteLists:
    if db.session is None:
        raise Responses_500.database_not_initialized
    async with db.session() as transaction:
        whitelist_count_stmt = select(func.count(ApiWhiteLists.id)).filter_by(api_key_id=api_key_id)
        whitelist_count: int | None = await transaction.scalar(whitelist_count_stmt)
        if whitelist_count is not None and whitelist_count >= MAX_API_WHITELIST:
            raise Responses_400.max_whitekey_count_exceed
        ip_duplicated_whitelist_stmt = select(ApiWhiteLists).filter_by(api_key_id=api_key_id, ip_address=ip_address)
        ip_duplicated_whitelist = await transaction.scalar(ip_duplicated_whitelist_stmt)
        if ip_duplicated_whitelist is not None:
            return ip_duplicated_whitelist
        new_whitelist = ApiWhiteLists(api_key_id=api_key_id, ip_address=ip_address)
        transaction.add(new_whitelist)
        await transaction.commit()
        await transaction.refresh(new_whitelist)
        return new_whitelist


async def get_api_key_whitelist(api_key_id: int) -> list[ApiWhiteLists]:
    return await ApiWhiteLists.fetchall_filtered_by(api_key_id=api_key_id)


async def delete_api_key_whitelist(
    user_id: int,
    api_key_id: int,
    whitelist_id: int,
) -> None:
    if db.session is None:
        raise Responses_500.database_not_initialized
    async with db.session() as transaction:
        matched_api_key: Optional[ApiKeys] = await transaction.scalar(
            select(ApiKeys).filter_by(id=api_key_id, user_id=user_id)
        )
        if matched_api_key is None:
            raise Responses_404.not_found_api_key
        matched_whitelist_stmt = select(ApiWhiteLists).filter_by(id=whitelist_id, api_key_id=api_key_id)
        matched_whitelist = await transaction.scalar(matched_whitelist_stmt)
        await transaction.delete(matched_whitelist)
        await transaction.commit()
