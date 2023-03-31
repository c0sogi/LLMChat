from typing import Optional, Tuple
from sqlalchemy import select, func, exists
from app.models.base_models import AddApiKey
from app.errors.api_exceptions import (
    Responses_400,
    Responses_404,
)
from app.common.config import MAX_API_KEY, MAX_API_WHITELIST
from app.database.schema import (
    ChatMessages,
    ChatRooms,
    GptPresets,
    db,
    Users,
    ApiKeys,
    ApiWhiteLists,
)
from app.utils.encoding_and_hashing import generate_api_key


async def is_email_exist(email: str) -> bool:
    return True if await db.scalar(select(exists().where(Users.email == email))) else False


async def get_me(user_id: int):
    return await Users.first_filtered_by(id=user_id)


async def get_api_key_and_owner(access_key: str) -> Tuple[ApiKeys, Users]:
    async with db.session() as transaction:
        matched_api_key: Optional[ApiKeys] = await transaction.scalar(select(ApiKeys).filter_by(access_key=access_key))
        if matched_api_key is None:
            raise Responses_404.not_found_access_key
        api_key_owner: Optional[Users] = await transaction.scalar(select(Users).filter_by(id=matched_api_key.user_id))
        if api_key_owner is None:
            raise Responses_404.not_found_user
        return matched_api_key, api_key_owner


async def is_valid_api_key(access_key: str) -> bool:
    return True if await db.scalar(select(exists().where(ApiKeys.access_key == access_key))) else False


async def get_api_key_owner(access_key: str) -> Users:
    async with db.session() as transaction:
        matched_api_key: Optional[ApiKeys] = await transaction.scalar(select(ApiKeys).filter_by(access_key=access_key))
        if matched_api_key is None:
            raise Responses_404.not_found_access_key
        owner: Users = await Users.first_filtered_by(id=matched_api_key.user_id)
        if owner is None:
            raise Responses_404.not_found_user
        return owner


async def register_new_user(
    email: str,
    hashed_password: str,
    ip_address: str,
) -> Users:
    return await Users.add_one(
        autocommit=True,
        refresh=True,
        email=email,
        password=hashed_password,
        ip_address=ip_address,
    )


async def find_matched_user(email: str) -> Users:
    return await Users.first_filtered_by(email=email)


async def create_api_key(
    user_id: int,
    additional_key_info: AddApiKey,
) -> ApiKeys:
    async with db.session() as transaction:
        api_key_count_stmt = select(func.count(ApiKeys.id)).filter_by(user_id=user_id)
        api_key_count: int = await transaction.scalar(api_key_count_stmt)
        if api_key_count >= MAX_API_KEY:
            raise Responses_400.max_key_count_exceed
        while True:
            new_api_key: ApiKeys = await generate_api_key(user_id=user_id, additional_key_info=additional_key_info)
            is_api_key_duplicate_stmt = select(exists().where(ApiKeys.access_key == new_api_key.access_key))
            is_api_key_duplicate: bool = await transaction.scalar(is_api_key_duplicate_stmt)
            if not is_api_key_duplicate:
                break
        transaction.add(new_api_key)
        await transaction.commit()
        await transaction.refresh(new_api_key)
        return new_api_key


async def get_api_keys(user_id: int) -> list[ApiKeys]:
    return await ApiKeys.fetchall_filtered_by(user_id=user_id)


async def update_api_key(
    updated_key_info: dict,
    access_key_id: int,
    user_id: int,
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


async def delete_api_key(
    access_key_id: int,
    access_key: str,
    user_id: int,
) -> None:
    async with db.session() as transaction:
        matched_api_key: Optional[ApiKeys] = await transaction.scalar(
            select(ApiKeys).filter_by(id=access_key_id, user_id=user_id, access_key=access_key)
        )
        if matched_api_key is None:
            raise Responses_404.not_found_api_key
        await transaction.delete(matched_api_key)
        await transaction.commit()


async def create_api_key_whitelist(
    ip_address: str,
    api_key_id: int,
) -> ApiWhiteLists:
    async with db.session() as transaction:
        whitelist_count_stmt = select(func.count(ApiWhiteLists.id)).filter_by(api_key_id=api_key_id)
        whitelist_count: int = await transaction.scalar(whitelist_count_stmt)
        if whitelist_count >= MAX_API_WHITELIST:
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


async def get_api_key_whitelist(api_key_id: int) -> list[ApiKeys]:
    return await ApiWhiteLists.fetchall_filtered_by(api_key_id=api_key_id)


async def delete_api_key_whitelist(
    user_id: int,
    api_key_id: int,
    whitelist_id: int,
) -> None:
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


async def create_chat_room(
    chat_room_type: str,
    name: str,
    description: str | None,
    user_id: int,
    status: str = "active",
) -> ChatRooms:
    async with db.session() as transaction:
        new_chat_room: ChatRooms = ChatRooms(
            status=status,
            chat_room_type=chat_room_type,
            name=name,
            description=description,
            user_id=user_id,
        )
        transaction.add(new_chat_room)
        await transaction.commit()
        await transaction.refresh(new_chat_room)
        return new_chat_room


async def get_chat_all_rooms(user_id: int) -> list[ChatRooms]:
    return await ChatRooms.fetchall_filtered_by(user_id=user_id)


async def create_chat_message(
    role: str,
    message: str,
    chat_room_id: int,
    user_id: int,
    status: str = "active",
) -> ChatMessages:
    async with db.session() as transaction:
        new_chat_message: ChatMessages = ChatMessages(
            status=status,
            role=role,
            message=message,
            user_id=user_id,
            chat_room_id=chat_room_id,
        )
        transaction.add(new_chat_message)
        await transaction.commit()
        await transaction.refresh(new_chat_message)
        return new_chat_message


async def get_chat_all_messages(
    chat_room_id: int,
    user_id: int,
) -> list[ChatMessages]:
    return await ChatMessages.fetchall_filtered_by(user_id=user_id, chat_room_id=chat_room_id)


async def create_gpt_preset(
    user_id: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
) -> GptPresets:
    async with db.session() as transaction:
        new_gpt_preset: GptPresets = GptPresets(
            user_id=user_id,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        transaction.add(new_gpt_preset)
        await transaction.commit()
        await transaction.refresh(new_gpt_preset)
        return new_gpt_preset


async def get_gpt_presets(user_id: int) -> list[GptPresets]:
    return await GptPresets.fetchall_filtered_by(user_id=user_id)


async def get_gpt_preset(user_id: int, preset_id: int) -> GptPresets:
    return await GptPresets.fetchone_filtered_by(user_id=user_id, id=preset_id)


async def update_gpt_preset(
    user_id: int,
    preset_id: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    status: str = "active",
) -> GptPresets:
    async with db.session() as transaction:
        matched_preset: Optional[GptPresets] = await transaction.scalar(
            select(GptPresets).filter_by(user_id=user_id, id=preset_id)
        )
        if matched_preset is None:
            raise Responses_404.not_found_preset
        matched_preset.set_values_as(
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            status=status,
        )
        transaction.add(matched_preset)
        await transaction.commit()
        await transaction.refresh(matched_preset)
        return matched_preset


async def delete_gpt_preset(user_id: int, preset_id: int) -> None:
    async with db.session() as transaction:
        matched_preset: Optional[GptPresets] = await transaction.scalar(
            select(GptPresets).filter_by(user_id=user_id, id=preset_id)
        )
        if matched_preset is None:
            raise Responses_404.not_found_preset
        await transaction.delete(matched_preset)
        await transaction.commit()
        return
