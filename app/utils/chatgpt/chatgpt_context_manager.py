import json
from app.database.connection import RedisFactory, cache
from app.viewmodels.gpt_models import GPT_MODELS, MessageHistory, UserGptContext, UserGptProfile


class ChatGptContextManager:
    _context_fields: tuple[str] = (
        "user_gpt_profile",
        "gpt_model",
        "user_message_histories",
        "gpt_message_histories",
        "system_message_histories",
    )
    _string_fields: tuple[str] = (
        "user_gpt_profile",
        "gpt_model",
    )
    _list_fields: tuple[str] = (
        "user_message_histories",
        "gpt_message_histories",
        "system_message_histories",
    )

    def __init__(self, cache: RedisFactory):
        self.cache = cache

    def _generate_key(self, user_id: int, field: str) -> str:
        return f"chatgpt:user:{user_id}:{field}"

    def _get_string_fields(self, user_id: int) -> dict[str, str]:
        return {field: self._generate_key(user_id, field) for field in self._string_fields}

    def _get_list_fields(self, user_id: int) -> dict[str, str]:
        return {field: self._generate_key(user_id, field) for field in self._list_fields}

    def read_context(self, user_id: int) -> UserGptContext:
        stored_string: dict[str, str | None] = {
            field: self.cache.redis.get(key) for field, key in self._get_string_fields(user_id).items()
        }
        stored_list: dict[str, list[str] | None] = {
            field: self.cache.redis.lrange(key, 0, -1) for field, key in self._get_list_fields(user_id).items()
        }

        # if any of stored strings are None, create new context
        if any([value is None for value in stored_string.values()]):
            default: UserGptContext = UserGptContext.construct_default(user_id=user_id)
            self.create_context(user_id=user_id, user_gpt_context=default)
            return default

        for field, value in stored_string.items():
            if value is not None:
                stored_string[field] = json.loads(value)
        for field, value in stored_list.items():
            if value is not None:
                stored_list[field] = [json.loads(v) for v in value]

        user_message_histories = [MessageHistory(**m) for m in stored_list["user_message_histories"]]
        gpt_message_histories = [MessageHistory(**m) for m in stored_list["gpt_message_histories"]]
        system_message_histories = [MessageHistory(**m) for m in stored_list["system_message_histories"]]
        return UserGptContext(
            user_gpt_profile=UserGptProfile(**stored_string["user_gpt_profile"]),
            gpt_model=getattr(GPT_MODELS, stored_string["gpt_model"].replace(".", "_").replace("-", "_")),
            user_message_histories=user_message_histories,
            gpt_message_histories=gpt_message_histories,
            system_message_histories=system_message_histories,
            user_message_tokens=sum([m.tokens for m in user_message_histories]),
            gpt_message_tokens=sum([m.tokens for m in gpt_message_histories]),
            system_message_tokens=sum([m.tokens for m in system_message_histories]),
            is_discontinued=False,
        )

    def create_context(
        self,
        user_id: int,
        user_gpt_context: UserGptContext,
        only_if_exists: bool = False,
        only_if_not_exists: bool = True,
    ) -> bool:
        json_data = user_gpt_context.json()
        success: bool = True

        for field, key in self._get_string_fields(user_id=user_id).items():
            result = self.cache.redis.set(
                key,
                json.dumps(json_data[field]),
                xx=only_if_exists,
                nx=only_if_not_exists,
            )
            success &= bool(result)
        for field, key in self._get_list_fields(user_id=user_id).items():
            result = self.cache.redis.delete(key)
            success &= bool(result)
            for item in json_data[field]:
                result = self.cache.redis.rpush(key, json.dumps(item))
                success &= bool(result)

        return success

    def reset_context(
        self,
        user_id: int,
        only_if_exists: bool = True,
        only_if_not_exists: bool = False,
    ) -> bool:
        return self.create_context(
            user_id=user_id,
            user_gpt_context=UserGptContext.construct_default(user_id=user_id),
            only_if_exists=only_if_exists,
            only_if_not_exists=only_if_not_exists,
        )

    def update_context(self, user_gpt_context: UserGptContext, only_if_exists: bool = True) -> bool:
        json_data = user_gpt_context.json()
        success = True

        for field, key in self._get_string_fields(user_id=user_gpt_context.user_gpt_profile.user_id).items():
            result = self.cache.redis.set(
                key,
                json.dumps(json_data[field]),
                xx=only_if_exists,
            )
            success &= bool(result)
        for field, key in self._get_list_fields(user_id=user_gpt_context.user_gpt_profile.user_id).items():
            result = self.cache.redis.delete(key)
            success &= bool(result)
            for item in json_data[field]:
                result = self.cache.redis.rpush(key, json.dumps(item))
                success &= bool(result)
        return success

    def update_message_histories(self, user_id: int, role: str, message_histories: list[MessageHistory]) -> bool:
        key = self._generate_key(user_id, f"{role}_message_histories")
        message_histories_json = [json.dumps(message_history.__dict__) for message_history in message_histories]
        result = self.cache.redis.delete(key)
        result &= self.cache.redis.rpush(key, *message_histories_json)
        return bool(result)

    def append_message_history(
        self,
        user_id: int,
        role: str,
        message_history: MessageHistory,
        if_exists: bool = False,
    ) -> bool:
        message_history_key = self._generate_key(user_id, f"{role}_message_histories")
        message_history_json = json.dumps(message_history.__dict__)
        result = (
            self.cache.redis.rpush(message_history_key, message_history_json)
            if not if_exists
            else self.cache.redis.rpushx(message_history_key, message_history_json)
        )
        return bool(result)

    def get_message_history(self, user_id: int, role: str) -> list[MessageHistory]:
        key = self._generate_key(user_id, f"{role}_message_histories")
        raw_message_histories = self.cache.redis.lrange(key, 0, -1)
        if raw_message_histories is None:
            return []
        return [MessageHistory(**json.loads(raw_message_history)) for raw_message_history in raw_message_histories]

    def delete_message_history(self, user_id: int, role: str) -> bool:
        key = self._generate_key(user_id, f"{role}_message_histories")
        result = self.cache.redis.delete(key)
        return bool(result)

    def update_tokens(self, user_id: int, role: str, set_tokens_as: int, if_exists: bool = True) -> bool:
        key = self._generate_key(user_id, f"{role}_message_tokens")
        result = self.cache.redis.set(key, set_tokens_as, xx=if_exists)
        return bool(result)

    def get_tokens(self, user_id: int, role: str) -> int:
        key = self._generate_key(user_id, f"{role}_message_tokens")
        tokens = self.cache.redis.get(key)
        return int(tokens) if tokens is not None else 0

    def delete_tokens(self, user_id: int, role: str) -> bool:
        key = self._generate_key(user_id, f"{role}_message_tokens")
        result = self.cache.redis.delete(key)
        return bool(result)


context_manager: ChatGptContextManager = ChatGptContextManager(cache=cache)
