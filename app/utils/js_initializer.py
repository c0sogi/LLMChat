import re
from os import environ


def replace_string(file_path: str, before: str, after: str) -> None:
    with open(file=file_path, mode="r") as file:
        filedata = file.read()

    with open(file=file_path, mode="w") as file:
        file.write(re.sub(before, after, filedata))


def js_url_initializer(js_location: str = "app/web/main.dart.js") -> None:
    fallback_host: str = "localhost:8000"
    fallback_env: str = "local"
    HOST_MAIN: str = environ.get("HOST_MAIN", fallback_host)
    env: str = environ.get("API_ENV", fallback_env)

    replace_string(
        file_path=js_location,
        before=f"http://{fallback_host}" if env == "prod" else f"https://{HOST_MAIN}",
        after=f"https://{HOST_MAIN}" if env == "prod" else f"http://{fallback_host}",
    )
    replace_string(
        file_path=js_location,
        before="ws://localhost:8000" if env == "prod" else f"wss://{HOST_MAIN}",
        after=f"wss://{HOST_MAIN}" if env == "prod" else f"ws://{fallback_host}",
    )
