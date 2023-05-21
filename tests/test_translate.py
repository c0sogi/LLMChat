import pytest
from app.common.config import (
    GOOGLE_TRANSLATE_API_KEY,
    PAPAGO_CLIENT_ID,
    PAPAGO_CLIENT_SECRET,
    RAPID_API_KEY,
    CUSTOM_TRANSLATE_URL,
)
from app.utils.api.translate import Translator


@pytest.mark.asyncio
async def test_custom_translate(test_logger):
    translated = await Translator.custom_translate_api(
        text="hello",
        src_lang="en",
        trg_lang="ko",
        api_url=str(CUSTOM_TRANSLATE_URL),
    )
    test_logger.info(__name__ + ":" + translated)


@pytest.mark.asyncio
async def test_google_translate(test_logger):
    translated = await Translator.google(
        text="hello",
        src_lang="en",
        trg_lang="ko",
        api_key=str(GOOGLE_TRANSLATE_API_KEY),
    )
    test_logger.info(__name__ + ":" + translated)


@pytest.mark.asyncio
async def test_papago(test_logger):
    translated = await Translator.papago(
        text="hello",
        src_lang="en",
        trg_lang="ko",
        client_id=str(PAPAGO_CLIENT_ID),
        client_secret=str(PAPAGO_CLIENT_SECRET),
    )
    test_logger.info(__name__ + ":" + translated)


@pytest.mark.asyncio
async def test_deepl(test_logger):
    translated = await Translator.deepl_via_rapid_api(
        text="hello",
        src_lang="en",
        trg_lang="ko",
        api_key=str(RAPID_API_KEY),
    )
    test_logger.info(__name__ + ":" + translated)


@pytest.mark.asyncio
async def test_auto_translate(test_logger):
    translated = await Translator.translate(
        text="hello",
        src_lang="en",
        trg_lang="ko",
    )
    test_logger.info("FIRST:" + translated + str(Translator.cached_function) + str(Translator.cached_args))
    translated = await Translator.translate(
        text="what is your name?",
        src_lang="en",
        trg_lang="ko",
    )
    test_logger.info("SECOND:" + translated + str(Translator.cached_function) + str(Translator.cached_args))
