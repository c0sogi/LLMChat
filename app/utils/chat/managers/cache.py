from asyncio import gather
from dataclasses import asdict, fields
from uuid import uuid4

from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads

from app.common.config import DEFAULT_LLM_MODEL
from app.database.connection import cache
from app.models.chat_models import (
    ChatRoles,
    MessageHistory,
    UserChatContext,
    UserChatProfile,
)
from app.models.llms import LLMModels
from app.utils.logger import ApiLogger


class CacheManager:
    _string_fields: tuple = (
        "user_chat_profile",
        "llm_model",
    )
    _list_fields: tuple[str] = tuple(
        f"{role.name.lower()}_message_histories" for role in ChatRoles
    )

    # Helper methods
    @staticmethod
    def _generate_key(user_id: str, chat_room_id: str, field: str) -> str:
        return f"chat:{user_id}:{chat_room_id}:{field}"

    @classmethod
    def _get_string_fields(
        cls, user_id: str, chat_room_id: str
    ) -> dict[str, str]:
        return {
            field: cls._generate_key(user_id, chat_room_id, field)
            for field in cls._string_fields
        }

    @classmethod
    def _get_list_fields(
        cls, user_id: str, chat_room_id: str
    ) -> dict[str, str]:
        return {
            field: cls._generate_key(user_id, chat_room_id, field)
            for field in cls._list_fields
        }

    @classmethod
    async def _load_chat_profile(cls, key: bytes) -> UserChatProfile | None:
        """Loads chat profile from redis."""
        try:
            loaded = await cache.redis.get(key)
            assert loaded is not None
            kwargs = {
                k: v
                for k, v in orjson_loads(loaded).items()
                if any(f.name == k for f in fields(UserChatProfile))
            }
            return UserChatProfile(**kwargs)
        except Exception:
            ApiLogger.cerror("Error reading profile from cache", exc_info=True)
            return None

    # Public methods
    @classmethod
    async def fetch_chat_profiles(cls, user_id: str) -> list[UserChatProfile]:
        """Fetches chat profile values from redis."""
        keys: list[bytes] = []
        cursor: int = 0  # Start with cursor 0.
        while True:
            cursor, batch_keys = await cache.redis.scan(
                cursor, match=f"chat:{user_id}:*:user_chat_profile"
            )
            keys.extend(batch_keys)
            if cursor == 0:
                break
        return [
            profile
            for profile in await gather(
                *(cls._load_chat_profile(key) for key in keys)
            )
            if profile is not None
        ]

    @classmethod
    async def read_context_from_profile(
        cls, user_chat_profile: UserChatProfile
    ) -> UserChatContext:
        user_id: str = user_chat_profile.user_id
        chat_room_id: str = user_chat_profile.chat_room_id

        stored_string: dict[str, str | None] = {
            field: await cache.redis.get(key)
            for field, key in cls._get_string_fields(
                user_id, chat_room_id
            ).items()
            if field != "user_chat_profile"
        }
        stored_list: dict[str, list | None] = {
            field: await cache.redis.lrange(key, 0, -1)
            for field, key in cls._get_list_fields(
                user_id, chat_room_id
            ).items()
        }

        # if any of stored strings are None, create new context
        if any([value is None for value in stored_string.values()]):
            default: UserChatContext = UserChatContext.construct_default(
                user_id=user_id,
                chat_room_id=chat_room_id,
            )
            await cls.create_context(default)
            return default
        try:
            for field, value in stored_string.items():
                if value is not None:
                    stored_string[field] = orjson_loads(value)
            for field, value in stored_list.items():
                if value is not None:
                    stored_list[field] = [orjson_loads(v) for v in value]
            if stored_string["llm_model"] not in LLMModels.member_names:
                stored_string["llm_model"] = DEFAULT_LLM_MODEL
            return UserChatContext(
                user_chat_profile=user_chat_profile,
                llm_model=LLMModels.get_member(
                    stored_string["llm_model"],
                ),
                user_message_histories=[
                    MessageHistory(**m)
                    for m in stored_list["user_message_histories"]
                    if m["content"]
                ]
                if stored_list["user_message_histories"] is not None
                else [],
                ai_message_histories=[
                    MessageHistory(**m)
                    for m in stored_list["ai_message_histories"]
                    if m["content"]
                ]
                if stored_list["ai_message_histories"] is not None
                else [],
                system_message_histories=[
                    MessageHistory(**m)
                    for m in stored_list["system_message_histories"]
                    if m["content"]
                ]
                if stored_list["system_message_histories"] is not None
                else [],
            )
        except Exception:
            ApiLogger.cerror("Error reading context from cache", exc_info=True)
            default: UserChatContext = UserChatContext.construct_default(
                user_id=user_id,
                chat_room_id=chat_room_id,
            )
            await cls.create_context(default)
            return default

    @classmethod
    async def create_context(
        cls,
        user_chat_context: UserChatContext,
        only_if_exists: bool = False,
        only_if_not_exists: bool = True,
    ) -> bool:
        json_data = user_chat_context.json()
        success: bool = True
        for field, key in cls._get_string_fields(
            user_chat_context.user_id, user_chat_context.chat_room_id
        ).items():
            result = await cache.redis.set(
                key,
                orjson_dumps(json_data[field]),
                xx=only_if_exists,
                nx=only_if_not_exists,
            )
            success &= bool(result)
        for field, key in cls._get_list_fields(
            user_chat_context.user_id, user_chat_context.chat_room_id
        ).items():
            result = await cache.redis.delete(key)
            for item in json_data[field]:
                result = await cache.redis.rpush(key, orjson_dumps(item))
                success &= bool(result)

        return success

    @classmethod
    async def reset_context(
        cls,
        user_id: str,
        chat_room_id: str,
        only_if_exists: bool = True,
        only_if_not_exists: bool = False,
    ) -> bool:
        return await cls.create_context(
            UserChatContext.construct_default(
                user_id=user_id,
                chat_room_id=chat_room_id,
            ),
            only_if_exists=only_if_exists,
            only_if_not_exists=only_if_not_exists,
        )

    @classmethod
    async def update_context(
        cls,
        user_chat_context: UserChatContext,
        only_if_exists: bool = True,
    ) -> bool:
        json_data = user_chat_context.json()
        success = True
        for field, key in cls._get_string_fields(
            user_chat_context.user_id,
            user_chat_context.chat_room_id,
        ).items():
            result = await cache.redis.set(
                key,
                orjson_dumps(json_data[field]),
                xx=only_if_exists,
            )
            success &= bool(result)
        for field, key in cls._get_list_fields(
            user_chat_context.user_id,
            user_chat_context.chat_room_id,
        ).items():
            result = await cache.redis.delete(key)
            success &= bool(result)
            for item in json_data[field]:
                result = await cache.redis.rpush(key, orjson_dumps(item))
                success &= bool(result)
        return success

    @classmethod
    async def delete_chat_room(cls, user_id: str, chat_room_id: str) -> int:
        # delete all keys starting with "chat:{user_id}:{chat_room_id}:"
        keys = [
            key
            async for key in cache.redis.scan_iter(
                f"chat:{user_id}:{chat_room_id}:*"
            )
        ]
        if not keys:
            return 0
        return await cache.redis.delete(*keys)

    @classmethod
    async def delete_user(cls, user_id: str) -> int:
        # delete all keys starting with "chat:{user_id}:"
        keys = [key async for key in cache.redis.scan_iter(f"*:{user_id}:*")]
        if not keys:
            return 0
        return await cache.redis.delete(*keys)

    @classmethod
    async def update_profile_and_model(
        cls,
        user_chat_context: UserChatContext,
        only_if_exists: bool = True,
    ) -> bool:
        json_data = user_chat_context.json()
        success = True

        for field, key in cls._get_string_fields(
            user_chat_context.user_id,
            user_chat_context.chat_room_id,
        ).items():
            result = await cache.redis.set(
                key,
                orjson_dumps(json_data[field]),
                xx=only_if_exists,
            )
            success &= bool(result)
        return success

    @classmethod
    async def update_profile(
        cls,
        user_chat_profile: UserChatProfile,
        only_if_exists: bool = True,
    ) -> bool:
        key: str = cls._generate_key(
            user_id=user_chat_profile.user_id,
            chat_room_id=user_chat_profile.chat_room_id,
            field="user_chat_profile",
        )

        return (
            await cache.redis.set(
                key,
                orjson_dumps(asdict(user_chat_profile)),
                xx=only_if_exists,
            )
        ) is True

    @classmethod
    async def update_model(
        cls,
        user_chat_context: UserChatContext,
        only_if_exists: bool = True,
    ) -> bool:
        key: str = cls._generate_key(
            user_id=user_chat_context.user_id,
            chat_room_id=user_chat_context.chat_room_id,
            field="llm_model",
        )

        return (
            await cache.redis.set(
                key,
                orjson_dumps(user_chat_context.llm_model.name),
                xx=only_if_exists,
            )
        ) is True

    @classmethod
    async def update_message_histories(
        cls,
        user_id: str,
        chat_room_id: str,
        role: ChatRoles,
        message_histories: list[MessageHistory],
    ) -> bool:
        field = f"{ChatRoles.get_name(role).lower()}_message_histories"
        key = cls._generate_key(user_id, chat_room_id, field)
        message_histories_json = [
            orjson_dumps(message_history.__dict__)
            for message_history in message_histories
        ]
        result = await cache.redis.delete(key)
        result &= await cache.redis.rpush(key, *message_histories_json)
        return bool(result)

    @classmethod
    async def lpop_message_history(
        cls,
        user_id: str,
        chat_room_id: str,
        role: ChatRoles,
        count: int | None = None,
    ) -> MessageHistory | list[MessageHistory] | None:
        assert count is None or count > 0
        field = f"{ChatRoles.get_name(role).lower()}_message_histories"
        message_history_json: str | list | None = await cache.redis.lpop(
            cls._generate_key(user_id, chat_room_id, field), count=count
        )
        if message_history_json is None:
            return None
        # if message_history_json is instance of list, then it is a list of message histories
        if isinstance(message_history_json, list):
            return [
                MessageHistory(**orjson_loads(m)) for m in message_history_json
            ]
        # otherwise, it is a single message history
        return MessageHistory(**orjson_loads(message_history_json))

    @classmethod
    async def rpop_message_history(
        cls,
        user_id: str,
        chat_room_id: str,
        role: ChatRoles,
        count: int | None = None,
    ) -> MessageHistory | list[MessageHistory] | None:
        field = f"{ChatRoles.get_name(role).lower()}_message_histories"
        assert count is None or count > 0
        message_history_json = await cache.redis.rpop(
            cls._generate_key(user_id, chat_room_id, field), count=count
        )
        if message_history_json is None:
            return None
        # if message_history_json is instance of list, then it is a list of message histories
        if isinstance(message_history_json, list):
            return [
                MessageHistory(**orjson_loads(m)) for m in message_history_json
            ]
        # otherwise, it is a single message history
        return MessageHistory(**orjson_loads(message_history_json))

    @classmethod
    async def append_message_history(
        cls,
        user_id: str,
        chat_room_id: str,
        role: ChatRoles,
        message_history: MessageHistory,
        if_exists: bool = False,
    ) -> bool:
        field = f"{ChatRoles.get_name(role).lower()}_message_histories"
        message_history_key = cls._generate_key(user_id, chat_room_id, field)
        message_history_json = orjson_dumps(message_history.__dict__)
        result = (
            await cache.redis.rpush(message_history_key, message_history_json)
            if not if_exists
            else await cache.redis.rpushx(
                message_history_key, message_history_json
            )
        )
        return bool(result)

    @classmethod
    async def get_message_history(
        cls,
        user_id: str,
        chat_room_id: str,
        role: ChatRoles,
    ) -> list[MessageHistory]:
        field = f"{ChatRoles.get_name(role).lower()}_message_histories"
        key = cls._generate_key(user_id, chat_room_id, field)
        raw_message_histories = await cache.redis.lrange(key, 0, -1)
        if raw_message_histories is None:
            return []
        return [
            MessageHistory(**orjson_loads(raw_message_history))
            for raw_message_history in raw_message_histories
        ]

    @classmethod
    async def delete_one_message_history(
        cls,
        user_id: str,
        chat_room_id: str,
        index: int,
        role: ChatRoles,
    ) -> bool:
        result: bool = True
        unique_id: str = uuid4().hex
        field = f"{ChatRoles.get_name(role).lower()}_message_histories"
        key = cls._generate_key(user_id, chat_room_id, field)
        # Set the value at index to the unique ID
        result &= await cache.redis.lset(key, index=index, value=unique_id)
        # Remove the unique ID from the list
        result &= bool(await cache.redis.lrem(key, count=1, value=unique_id))
        return result

    @classmethod
    async def delete_message_history(
        cls,
        user_id: str,
        chat_room_id: str,
        role: ChatRoles,
    ) -> bool:
        field = f"{ChatRoles.get_name(role).lower()}_message_histories"
        key: str = cls._generate_key(
            user_id=user_id,
            chat_room_id=chat_room_id,
            field=field,
        )
        return bool(await cache.redis.delete(key))

    @classmethod
    async def delete_message_histories(
        cls,
        user_id: str,
        chat_room_id: str,
        roles: list[ChatRoles],
    ) -> bool:
        keys: list[str] = [
            cls._generate_key(
                user_id=user_id,
                chat_room_id=chat_room_id,
                field=f"{ChatRoles.get_name(role).lower()}_message_histories",
            )
            for role in roles
        ]
        return bool(await cache.redis.delete(*keys))

    @classmethod
    async def set_message_history(
        cls,
        user_id: str,
        chat_room_id: str,
        message_history: MessageHistory,
        index: int,
        role: ChatRoles,
    ) -> bool:
        field = f"{ChatRoles.get_name(role).lower()}_message_histories"
        key = cls._generate_key(user_id, chat_room_id, field)
        result = await cache.redis.lset(
            key, index, orjson_dumps(message_history.__dict__)
        )
        return bool(result)
