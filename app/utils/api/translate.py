import httpx
import orjson
from app.common.config import GOOGLE_TRANSLATE_API_KEY, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET, RAPID_API_KEY


class Translator:
    @classmethod
    async def auto_translate(cls, text: str, src_lang: str, trg_lang: str) -> str:
        if RAPID_API_KEY is not None:
            try:
                return await cls.deeple_via_rapid_api(
                    text=text,
                    src_lang=src_lang,
                    trg_lang=trg_lang,
                    api_key=RAPID_API_KEY,
                )
            except Exception:
                ...
        if GOOGLE_TRANSLATE_API_KEY is not None:
            try:
                return await cls.google(
                    text=text,
                    src_lang=src_lang,
                    trg_lang=trg_lang,
                    api_key=GOOGLE_TRANSLATE_API_KEY,
                )
            except Exception:
                ...
        if PAPAGO_CLIENT_ID is not None and PAPAGO_CLIENT_SECRET is not None:
            try:
                return await cls.papago(
                    text=text,
                    src_lang=src_lang,
                    trg_lang=trg_lang,
                    client_id=PAPAGO_CLIENT_ID,
                    client_secret=PAPAGO_CLIENT_SECRET,
                )
            except Exception:
                ...
        return "번역 API가 설정되지 않았습니다."

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
        data = {"q": text, "source": src_lang, "target": trg_lang, "format": "text"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, data=data)
        return orjson.loads(response.text)["data"]["translations"][0]["translatedText"]

    @staticmethod
    async def deeple_via_rapid_api(
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
        content = orjson.dumps({"text": text, "source": src_lang, "target": trg_lang})
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, headers=headers, content=content)
        return orjson.loads(response.text)["text"]
