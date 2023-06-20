from sqlalchemy import select, exists
from app.database.connection import db
from app.database.schemas.auth import (
    Users,
    ApiKeys,
)


async def is_email_exist(email: str) -> bool:
    return True if await db.scalar(select(exists().where(Users.email == email))) else False


async def get_me(user_id: int):
    return await Users.first_filtered_by(id=user_id)


async def is_valid_api_key(access_key: str) -> bool:
    return True if await db.scalar(select(exists().where(ApiKeys.access_key == access_key))) else False


async def register_new_user(
    email: str,
    hashed_password: str,
    ip_address: str | None,
) -> Users:
    return (
        await Users.add_one(
            autocommit=True,
            refresh=True,
            email=email,
            password=hashed_password,
            ip_address=ip_address,
        )
        if ip_address
        else await Users.add_one(
            autocommit=True,
            refresh=True,
            email=email,
            password=hashed_password,
        )
    )


async def find_matched_user(email: str) -> Users:
    return await Users.first_filtered_by(email=email)
