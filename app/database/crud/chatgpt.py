from typing import Optional

from sqlalchemy import select
from app.errors.api_exceptions import (
    Responses_404,
    Responses_500,
)
from app.database.connection import db
from app.database.schemas.auth import (
    ChatMessages,
    ChatRooms,
    GptPresets,
)


async def create_chat_room(
    chat_room_type: str,
    name: str,
    description: str | None,
    user_id: int,
    status: str = "active",
) -> ChatRooms:
    if db.session is None:
        raise Responses_500.database_not_initialized
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
    return await ChatRooms.fetchall_filtered_by(user_id=user_id)  # type: ignore


async def create_chat_message(
    role: str,
    message: str,
    chat_room_id: int,
    user_id: int,
    status: str = "active",
) -> ChatMessages:
    if db.session is None:
        raise Responses_500.database_not_initialized
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
    return await ChatMessages.fetchall_filtered_by(
        user_id=user_id,
        chat_room_id=chat_room_id,
    )  # type: ignore


async def create_gpt_preset(
    user_id: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
) -> GptPresets:
    if db.session is None:
        raise Responses_500.database_not_initialized
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
    return await GptPresets.fetchall_filtered_by(user_id=user_id)  # type: ignore


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
    if db.session is None:
        raise Responses_500.database_not_initialized
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
    if db.session is None:
        raise Responses_500.database_not_initialized
    async with db.session() as transaction:
        matched_preset: Optional[GptPresets] = await transaction.scalar(
            select(GptPresets).filter_by(user_id=user_id, id=preset_id)
        )
        if matched_preset is None:
            raise Responses_404.not_found_preset
        await transaction.delete(matched_preset)
        await transaction.commit()
        return
