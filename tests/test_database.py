from app.models import AddApiKey, UserToken
from app.database.schema import Users, ApiKeys
from app.database.crud import create_api_key, get_api_key_and_owner
from app.middlewares.token_validator import validate_access_key
from app.utils.date_utils import UTC
from app.utils.query_utils import parse_params
from app.utils.encoding_and_hashing import hash_params
import pytest


@pytest.mark.asyncio
async def test_apikey_idenfitication(random_user):
    user: Users = await Users.add_one(autocommit=True, refresh=True, **random_user)
    additional_key_info: AddApiKey = AddApiKey(user_memo="[Testing] test_apikey_query")
    new_api_key: ApiKeys = await create_api_key(
        user_id=user.id, additional_key_info=additional_key_info
    )
    matched_api_key, matched_user = await get_api_key_and_owner(
        access_key=new_api_key.access_key
    )
    assert matched_api_key.access_key == new_api_key.access_key
    assert random_user["email"] == matched_user.email


@pytest.mark.asyncio
async def test_apikey_query(random_user):
    user: Users = await Users.add_one(autocommit=True, refresh=True, **random_user)
    additional_key_info: AddApiKey = AddApiKey(user_memo="[Testing] test_apikey_query")
    new_api_key: ApiKeys = await create_api_key(
        user_id=user.id, additional_key_info=additional_key_info
    )
    timestamp: str = str(UTC.timestamp(hour_diff=9))
    parsed_qs: str = parse_params(
        params={"key": new_api_key.access_key, "timestamp": timestamp}
    )
    user_token: UserToken = await validate_access_key(
        access_key=new_api_key.access_key,
        query_from_session=True,
        query_check=True,
        secret=hash_params(qs=parsed_qs, secret_key=new_api_key.secret_key),
        query_params=parsed_qs,
        timestamp=timestamp,
    )
    assert user_token.id == user.id
