import re
from app.common.config import config, ProdConfig


def js_url_initializer(js_location: str = "app/web/main.dart.js") -> None:
    with open(file=js_location, mode="r") as file:
        filedata = file.read()

    for schema in ("http", "ws"):
        if isinstance(config, ProdConfig):
            from_url_local = rf"{schema}://localhost:\d+"
            to_url_prod = f"{schema}s://{config.host_main}"
            filedata = re.sub(
                from_url_local,
                to_url_prod,
                filedata,
            )
        else:
            from_url_local = rf"{schema}://localhost:\d+"
            from_url_prod = rf"{schema}://{config.host_main}"
            to_url_local = f"{schema}://localhost:{config.port}"
            filedata = re.sub(
                from_url_local,
                to_url_local,
                filedata,
            )
            filedata = re.sub(
                from_url_prod,
                to_url_local,
                filedata,
            )

    with open(file=js_location, mode="w") as file:
        file.write(filedata)


if __name__ == "__main__":
    js_url_initializer()
