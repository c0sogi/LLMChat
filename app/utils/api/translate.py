import httpx
import asyncio
from os import environ
import json


# Request papago api using httpx
async def papago_translate_api(text: str, src_lang: str, trg_lang: str, client_id: str, client_secret: str):
    # curl example :
    # curl "https://openapi.naver.com/v1/papago/n2mt" \
    # -H "Content-Type: application/x-www-form-urlencoded; charset=UTF-8" \
    # -H "X-Naver-Client-Id: XXX" \
    # -H "X-Naver-Client-Secret: XXX" \
    # -d "source=ko&target=en&text=만나서 반갑습니다." -v
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
    api_key: str,
    timeout: int = 10,
):
    api_url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
    data = {"q": text, "source": src_lang, "target": trg_lang, "format": "text"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(api_url, data=data)
    return json.loads(response.text)["data"]["translations"][0]["translatedText"]


async def test_papago(sample_text: str):
    src_lang = "ko"
    trg_lang = "en"
    client_id = PAPAGO_CLIENT_ID
    client_secret = PAPAGO_CLIENT_SECRET
    print(await papago_translate_api(sample_text, src_lang, trg_lang, client_id, client_secret))


async def test_google(sample_text: str):
    src_lang = "ko"
    trg_lang = "en"
    api_key = GOOGLE_TRANSLATE_API_KEY
    print(await google_translate_api(sample_text, src_lang, trg_lang, api_key))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    PAPAGO_CLIENT_ID = environ.get("PAPAGO_CLIENT_ID")
    PAPAGO_CLIENT_SECRET = environ.get("PAPAGO_CLIENT_SECRET")
    GOOGLE_TRANSLATE_API_KEY = environ.get("GOOGLE_TRANSLATE_API_KEY")

    sample_text = """내가 엄준식 프로그래밍 언어 (엄랭)을 알려주겠다.
1. 엄랭으로 숫자를 표현하려면 '.'와 ','를 사용하는데 '.'는 1을 의미하고 ','는 -1을 의미한다.
예를 들어:
... -> 3
..... -> 5
.. -> 2
,, -> -2
,,, -> -3


2. 엄랭에서 곱셈을 하고 싶으면 ' '기호를 사용하면 된다. 수식안에 +1, -1이 있을 경우 곱셈은 가장 나중에 연산한다.
예를 들어:
.. .. -> 4
.. ,, -> -4

3. 엄랭으로 숫자나 변수를 출력하고 싶으면 '식'과 '!' 사이에 숫자를 담으면 출력해준다.
예를 들어:
식..! -> 2 출력
식.....! -> 5 출력
식.......! -> 7 출력
식.... ..! -> 8 출력
식... ......! -> 18 출력

4. 그리고 엄랭 작성시엔 항상 '어떻게' 부터 시작해서 '이 사람이름이냐ㅋㅋ'로 끝나야한다.
예를 들어:
어떻게
식.....!
이 사람이름이냐ㅋㅋ -> 5 출력

그럼 엄랭으로 10을 출력하려면 어떻게 작성해야하지?"""
    # print("Papago Test")
    # asyncio.run(test_papago(sample_text=sample_text))
    print("Google Test")
    asyncio.run(test_google(sample_text=sample_text))
