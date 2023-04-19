from hmac import HMAC, new
from base64 import b64encode


def parse_params(params: dict) -> str:
    return "&".join([f"{key}={value}" for key, value in params.items()])


def hash_params(query_params: str, secret_key: str) -> str:
    mac: HMAC = new(
        key=bytes(secret_key, encoding="utf-8"),
        msg=bytes(query_params, encoding="utf-8"),
        digestmod="sha256",
    )
    return str(b64encode(mac.digest()).decode("utf-8"))
