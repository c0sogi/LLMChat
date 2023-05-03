import httpx
import asyncio
from os import environ
import orjson
from app.common.config import GOOGLE_TRANSLATE_API_KEY, PAPAGO_CLIENT_ID, PAPAGO_CLIENT_SECRET

papago_client_id: str = "" if PAPAGO_CLIENT_ID is None else PAPAGO_CLIENT_ID
papago_client_secret: str = "" if PAPAGO_CLIENT_SECRET is None else PAPAGO_CLIENT_SECRET
google_translate_api_key: str = "" if GOOGLE_TRANSLATE_API_KEY is None else GOOGLE_TRANSLATE_API_KEY


# Request papago api using httpx
async def papago_translate_api(
    text: str,
    src_lang: str,
    trg_lang: str,
    client_id: str = papago_client_id,
    client_secret: str = papago_client_secret,
):
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


async def google_translate_api(
    text: str,
    src_lang: str,
    trg_lang: str,
    api_key: str = google_translate_api_key,
    timeout: int = 10,
):
    api_url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
    data = {"q": text, "source": src_lang, "target": trg_lang, "format": "text"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(api_url, data=data)
    return orjson.loads(response.text)["data"]["translations"][0]["translatedText"]
