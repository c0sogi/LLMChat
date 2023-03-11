from typing import Optional, Tuple
from sqlalchemy import select, func, exists
from app.models import AddApiKey
from app.errors import exceptions as exc
from app.common.config import MAX_API_KEY
from app.database.schema import db, Users, ApiKeys
from app.utils.encoding_and_hashing import generate_api_key


async def is_email_exist(email: str) -> bool:
    return (
        True if await db.scalar(select(exists().where(Users.email == email))) else False
    )


async def get_api_key_and_owner(access_key: str) -> Tuple[ApiKeys, Users]:
    async with db.session() as transaction:
        matched_api_key: Optional[ApiKeys] = await transaction.scalar(
            select(ApiKeys).filter_by(access_key=access_key)
        )
        if matched_api_key is None:
            raise exc.NotFoundAccessKeyEx(api_key=access_key)
        api_key_owner: Optional[Users] = await transaction.scalar(
            select(Users).filter_by(id=matched_api_key.user_id)
        )
        if api_key_owner is None:
            raise exc.NotFoundUserEx(user_id=matched_api_key.user_id)
        return matched_api_key, api_key_owner


async def create_new_api_key(user_id: int, additional_key_info: AddApiKey) -> ApiKeys:
    async with db.session() as transaction:
        api_key_count_stmt = select(func.count(ApiKeys.id)).filter_by(user_id=user_id)
        if (await transaction.scalar(api_key_count_stmt)) >= MAX_API_KEY:
            raise exc.MaxKeyCountEx()
        while True:
            new_api_key: ApiKeys = await generate_api_key(
                user_id=user_id, additional_key_info=additional_key_info
            )
            api_key_duplicate_check_stmt = select(
                exists().where(ApiKeys.access_key == new_api_key.access_key)
            )
            if not (await transaction.scalar(api_key_duplicate_check_stmt)):
                break
        transaction.add(new_api_key)
        await transaction.commit()
        await transaction.refresh(new_api_key)
    return new_api_key


async def get_all_api_keys(user_id: int):
    return await ApiKeys.fetchall_filtered_by(user_id=user_id)


async def register_new_user(email: str, hashed_password: str, ip_address: str):
    return await Users.add_one(
        autocommit=True,
        refresh=True,
        email=email,
        password=hashed_password,
        ip_address=ip_address,
    )


async def find_matched_user(email: str) -> Users:
    return await Users.first_filtered_by(email=email)
