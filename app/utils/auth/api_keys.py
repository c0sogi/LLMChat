from secrets import choice
from string import ascii_letters, digits
from uuid import uuid4

from app.database.schemas.auth import ApiKeys
from app.models.base_models import AddApiKey


def generate_new_api_key(
    user_id: int, additional_key_info: AddApiKey
) -> ApiKeys:
    alnums = ascii_letters + digits
    secret_key = "".join(choice(alnums) for _ in range(40))
    uid = f"{str(uuid4())[:-12]}{str(uuid4())}"
    new_api_key = ApiKeys(
        secret_key=secret_key,
        user_id=user_id,
        access_key=uid,
        **additional_key_info.dict(),
    )
    return new_api_key
