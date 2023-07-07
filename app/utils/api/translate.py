from typing import Callable
import httpx
import orjson
from app.utils.logger import ApiLogger
from app.common.config import (
    GOOGLE_TRANSLATE_API_KEY,
    PAPAGO_CLIENT_ID,
    PAPAGO_CLIENT_SECRET,
    RAPID_API_KEY,
    CUSTOM_TRANSLATE_URL,
)


class Translator:
    cached_function: Callable | None = None
    cached_args: dict = {}

    TRANSLATION_CONFIGS = [
        {
            "function": "custom_translate_api",
            "args": {"api_url": CUSTOM_TRANSLATE_URL},
        },
        {
            "function": "deepl_via_rapid_api",
            "args": {"api_key": RAPID_API_KEY},
        },
        {"function": "google", "args": {"api_key": GOOGLE_TRANSLATE_API_KEY}},
        {
            "function": "papago",
            "args": {
                "client_id": PAPAGO_CLIENT_ID,
                "client_secret": PAPAGO_CLIENT_SECRET,
            },
        },
    ]

    @classmethod
    async def translate(
        cls, text: str, src_lang: str, trg_lang: str = "en"
    ) -> str:
        if cls.cached_function is not None:
            try:
                ApiLogger.cinfo(
                    f"Using cached translate function: {cls.cached_function}"
                )
                return await cls.cached_function(
                    text=text,
                    src_lang=src_lang,
                    trg_lang=trg_lang,
                    **cls.cached_args,
                )
            except Exception:
                pass

        for cfg in cls.TRANSLATION_CONFIGS:
            function = getattr(cls, cfg["function"])
            args = cfg["args"]

            if all(arg is not None for arg in args.values()):
                try:
                    result = await function(
                        text=text,
                        src_lang=src_lang,
                        trg_lang=trg_lang,
                        **args,
                    )
                    ApiLogger.cinfo(
                        f"Succeeded to translate using {cfg['function']}"
                    )
                    cls.cached_function = function
                    cls.cached_args = args
                    return result
                except Exception:
                    ApiLogger.cerror(
                        f"Failed to translate using {cfg['function']}",
                        exc_info=True,
                    )
                    pass
        raise RuntimeError("Failed to translate")

    @staticmethod
    async def papago(
        text: str,
        src_lang: str,
        trg_lang: str,
        client_id: str,
        client_secret: str,
    ) -> str:
        api_url = "https://openapi.naver.com/v1/papago/n2mt"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }
        data = {"source": src_lang, "target": trg_lang, "text": text}
        # Request papago api using async httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, headers=headers, data=data)
            return response.json()["message"]["result"]["translatedText"]

    @staticmethod
    async def google(
        text: str,
        src_lang: str,
        trg_lang: str,
        api_key: str,
        timeout: int = 10,
    ) -> str:
        api_url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
        data = {
            "q": text,
            "source": src_lang,
            "target": trg_lang,
            "format": "text",
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, data=data)
        return orjson.loads(response.text)["data"]["translations"][0][
            "translatedText"
        ]

    @staticmethod
    async def deepl_via_rapid_api(
        text: str,
        src_lang: str,
        trg_lang: str,
        api_key: str,
        timeout: int = 10,
    ) -> str:
        api_host = "deepl-translator.p.rapidapi.com"
        api_url = f"https://{api_host}/translate"
        headers = {
            "Content-Type": "application/json",
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": api_host,
        }
        content = orjson.dumps(
            {
                "text": text,
                "source": src_lang.upper(),
                "target": trg_lang.upper(),
            }
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                api_url, headers=headers, content=content
            )
        return orjson.loads(response.text)["text"]

    @staticmethod
    async def custom_translate_api(
        text: str,
        src_lang: str,
        trg_lang: str,
        api_url: str,
        timeout: int = 10,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
        }
        content = orjson.dumps(
            {"text": text, "target_lang": trg_lang}
        )  # source_lang is excluded because we'll use auto-detection
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                api_url, headers=headers, content=content
            )
        return orjson.loads(response.text)["data"]
