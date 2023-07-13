import pytest

from app.common.config import OPENAI_API_KEY
from app.models.completion_models import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
)
from app.utils.api.completion import (
    acreate_chat_completion,
    acreate_completion,
    request_chat_completion,
    request_chat_completion_with_streaming,
    request_text_completion,
    request_text_completion_with_streaming,
)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OpenAI API Key is not set")
@pytest.mark.asyncio
async def test_completion_creation(stream: bool) -> None:
    assert OPENAI_API_KEY is not None

    prompt: str = "Say this is TEST:"
    model: str = "text-davinci-003"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = OPENAI_API_KEY
    max_tokens: int = 10
    temperature: float = 0
    n: int = 1

    async for chunk_or_completion in acreate_completion(
        prompt=prompt,
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stream=stream,
    ):
        if stream:
            chunk: CompletionChunk = chunk_or_completion  # type: ignore
            assert chunk["choices"][0]["text"] is not None
        else:
            completion: Completion = chunk_or_completion  # type: ignore
            assert completion["choices"][0]["text"] is not None


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OpenAI API Key is not set")
@pytest.mark.asyncio
async def test_chat_completion_creation(stream: bool) -> None:
    assert OPENAI_API_KEY is not None

    message: dict[str, str] = {"role": "user", "content": "Say this is TEST:"}
    model: str = "gpt-3.5-turbo"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = OPENAI_API_KEY
    max_tokens: int = 10
    temperature: float = 0
    n: int = 1

    async for chunk_or_completion in acreate_chat_completion(
        messages=[message],
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stream=stream,
    ):
        if stream:
            chunk: ChatCompletionChunk = chunk_or_completion  # type: ignore
            assert chunk["choices"][0] is not None
        else:
            completion: ChatCompletion = chunk_or_completion  # type: ignore
            assert completion["choices"][0] is not None


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OpenAI API Key is not set")
@pytest.mark.asyncio
async def test_text_completion_request(stream: bool) -> None:
    assert OPENAI_API_KEY is not None

    prompt: str = "Say this is TEST:"
    model: str = "text-davinci-003"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = OPENAI_API_KEY
    max_tokens: int = 10
    temperature: float = 0
    n: int = 1

    if stream:
        async for completion_chunk in request_text_completion_with_streaming(
            prompt=prompt,
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        ):
            assert completion_chunk["choices"][0]["text"] is not None
    else:
        completion: Completion = await request_text_completion(
            prompt=prompt,
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
        assert completion["choices"][0]["text"] is not None


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OpenAI API Key is not set")
@pytest.mark.asyncio
async def test_chat_completion_request(stream: bool) -> None:
    assert OPENAI_API_KEY is not None

    message: dict[str, str] = {"role": "user", "content": "Say this is TEST:"}
    model: str = "gpt-3.5-turbo"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = OPENAI_API_KEY
    max_tokens: int = 10
    temperature: float = 0
    n: int = 1
    if stream:
        async for chat_completion_chunk in request_chat_completion_with_streaming(
            messages=[message],
            model=model,
            api_base=api_base,
            api_key=api_key,
            functions=None,
            function_call=None,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        ):
            assert chat_completion_chunk["choices"][0] is not None
    else:
        chat_completion: ChatCompletion = await request_chat_completion(
            messages=[message],
            model=model,
            api_base=api_base,
            api_key=api_key,
            functions=None,
            function_call=None,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
        assert chat_completion["choices"][0]["message"]["content"] is not None
