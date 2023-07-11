import logging
from datetime import timedelta
from re import Pattern, compile
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

from aiohttp import ClientResponse, ClientSession
from openai import error
from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.common.config import ChatConfig
from app.errors.chat_exceptions import (
    ChatConnectionException,
    ChatTooMuchTokenException,
)
from app.models.completion_models import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
)
from app.models.function_calling.base import FunctionCall
from app.utils.chat.text_generations.converter import (
    make_chat_completion_chunk_from_json,
    make_chat_completion_from_json,
    make_completion_chunk_from_json,
    make_completion_from_json,
)
from app.utils.logger import ApiLogger

T = TypeVar("T")
TimeUnitType = Union[int, float, timedelta]
logger = logging.getLogger(__name__)
api_regex_pattern: Pattern = compile(
    r"data:\s*({.+?})\s*\r?\n\s*\r?\n"
)  # regex pattern to extract json from streaming api response


def _create_retry_decorator(
    max_retries: int = 6,
    exponential_base: Union[int, float] = 2,
    exponential_min_seconds: TimeUnitType = 1,
    exponential_max_seconds: TimeUnitType = 60,
    exponential_multiplier: Union[int, float] = 1,
) -> Callable[[Any], Any]:
    """Create retry decorator with given parameters
    Wait {exponential_base}^x * {exponential_multiplier} second
    between each retry starting with {exponential_min_seconds} and
    up to {exponential_max_seconds} seconds."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(
            multiplier=exponential_multiplier,
            exp_base=exponential_base,
            min=exponential_min_seconds,
            max=exponential_max_seconds,
        ),
        retry=(
            retry_if_exception_type(error.TryAgain)
            | retry_if_exception_type(error.RateLimitError)
            | retry_if_exception_type(error.Timeout)
            # | retry_if_exception_type(error.APIError)
            # | retry_if_exception_type(error.APIConnectionError)
            # | retry_if_exception_type(error.InvalidRequestError)
            # | retry_if_exception_type(error.PermissionError)
            # | retry_if_exception_type(error.ServiceUnavailableError)
            # | retry_if_exception_type(error.InvalidAPIType)
            # | retry_if_exception_type(error.SignatureVerificationError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _decode_function_call(
    functions: Optional[List[FunctionCall]] = None,
    function_call: Optional[FunctionCall | str] = None,
) -> Dict[str, Union[str, Dict[str, str]]]:
    """Decode functions & function_call into dict"""
    result: Dict[str, Any] = {}
    if functions is not None:
        result["functions"] = [function.to_dict() for function in functions]
    if function_call is not None:
        if isinstance(function_call, FunctionCall):
            result["function_call"] = {"name": function_call.to_dict()["name"]}
        elif isinstance(function_call, str):
            result["function_call"] = function_call
    return result


async def _extract_json_from_streaming_response(
    streaming_response: ClientResponse,
) -> AsyncIterator[dict]:
    """Extract json from streaming `aiohttp.ClientResponse`"""
    stream_buffer: bytes = b""
    async for stream, end_of_chunk in streaming_response.content.iter_chunks():  # stream from api
        stream_buffer += stream
        if not end_of_chunk:
            continue
        for match in api_regex_pattern.finditer(stream_buffer.decode("utf-8")):
            try:
                json_data: dict = orjson_loads(match.group(1))
                yield json_data
                stream_buffer = b""
            except Exception:
                continue


def _get_response_exception(
    rbody: str,
    rcode: int,
    resp: dict,
    rheaders: dict,
    stream_error: bool = False,
) -> error.OpenAIError:
    """Return appropriate error from response object"""
    try:
        error_data = resp["error"]
    except (KeyError, TypeError):
        raise error.APIError(
            "Invalid response object from API: %r (HTTP response code "
            "was %d)" % (rbody, rcode),
            rbody,
            rcode,
            resp,
        )

    if "internal_message" in error_data:
        error_data["message"] += "\n\n" + error_data["internal_message"]

    # Rate limits were previously coded as 400's with code 'rate_limit'
    if rcode == 429:
        return error.RateLimitError(
            error_data.get("message"), rbody, rcode, resp, rheaders
        )
    elif rcode in (400, 404, 415):
        return error.InvalidRequestError(
            error_data.get("message"),
            error_data.get("param"),
            error_data.get("code"),
            rbody,
            rcode,
            resp,
            rheaders,
        )
    elif rcode == 401:
        return error.AuthenticationError(
            error_data.get("message"), rbody, rcode, resp, rheaders
        )
    elif rcode == 403:
        return error.PermissionError(
            error_data.get("message"), rbody, rcode, resp, rheaders
        )
    elif rcode == 409:
        return error.TryAgain(
            error_data.get("message"), rbody, rcode, resp, rheaders
        )
    elif stream_error:
        # TODO: we will soon attach status codes to stream errors
        parts = [
            error_data.get("message"),
            "(Error occurred while streaming.)",
        ]
        message = " ".join([p for p in parts if p is not None])
        return error.APIError(message, rbody, rcode, resp, rheaders)
    else:
        return error.APIError(
            f"{error_data.get('message')} {rbody} {rcode} {resp} {rheaders}",
            rbody,
            rcode,
            resp,
            rheaders,
        )


async def _handle_error_response(
    response: ClientResponse,
) -> None:
    """Handle error from client response"""
    if not response.ok:  # if status code is not 200
        rbody: str = await response.text()
        rcode: int = response.status
        resp: dict = orjson_loads(rbody)
        rheaders: dict = dict(response.headers)
        error: Any = resp.get("error")
        response.release()

        ApiLogger("||agenerate_from_chat_completion_api||").error(
            f"API request Error: {error}"
        )
        if isinstance(error, dict):
            error_msg = str(error.get("message"))
            if "maximum context length" in error_msg:
                raise ChatTooMuchTokenException(msg="")
        else:
            error_msg = str(error)
        raise _get_response_exception(
            rbody, rcode, resp, rheaders, stream_error=True
        )


def _make_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Make headers to use for completion API"""
    if api_key is None:
        return {"Content-Type": "application/json"}
    else:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }


async def acompletion_with_retry(
    async_callback: Callable[..., Awaitable[T]],
    /,
    *args,
    max_retries: int = 6,
    **kwargs: Any,
) -> T:
    """Use tenacity to retry the async completion call."""

    @_create_retry_decorator(max_retries)
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await async_callback(*args, **kwargs)

    try:
        return await _completion_with_retry(*args, **kwargs)
    except error.OpenAIError as e:
        raise ChatConnectionException(
            msg=f"Completion API error: {e}"
        )  # raise exception for connection error


async def request_text_completion(
    prompt: str,
    model: str = ChatConfig.global_openai_model,
    api_base: str = "https://api.openai.com/v1",
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Completion:
    """Request text completion from API with proper retry logic.

    Args:
        prompt: The prompt to generate completions for.
        model_name: Name of the model to use. Defaults to ChatConfig.global_openai_model.
    """
    kwargs.pop("stream", None)

    async def get_text_completion(*args, **kwargs) -> Completion:
        async for text_completion in acreate_completion(*args, **kwargs):
            if TYPE_CHECKING:
                assert isinstance(text_completion, Completion)
            return text_completion
        raise ChatConnectionException(msg="Completion API error")

    return await acompletion_with_retry(
        get_text_completion,
        prompt=prompt,
        model=model,
        api_base=api_base,
        api_key=api_key,
        stream=False,
        **kwargs,
    )


async def request_text_completion_with_streaming(
    prompt: str,
    model: str = ChatConfig.global_openai_model,
    api_base: str = "https://api.openai.com/v1",
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> AsyncIterator[CompletionChunk]:
    """Request text completion streaming from API with proper retry logic.

    Args:
        prompt: The prompt to generate completions for.
        model_name: Name of the model to use. Defaults to ChatConfig.global_openai_model.
    """
    kwargs.pop("stream", None)

    async def get_text_completion_chunks(
        *args, **kwargs
    ) -> AsyncIterator[Completion | CompletionChunk]:
        return acreate_completion(*args, **kwargs)

    async for chunk in await acompletion_with_retry(
        get_text_completion_chunks,
        model=model,
        prompt=prompt,
        api_base=api_base,
        api_key=api_key,
        stream=True,
        **kwargs,
    ):
        if TYPE_CHECKING:
            assert isinstance(chunk, CompletionChunk)
        yield chunk


async def request_chat_completion(
    messages: List[Dict[str, str]],
    model: str = ChatConfig.global_openai_model,
    api_base: str = "https://api.openai.com/v1",
    api_key: Optional[str] = None,
    functions: Optional[List[FunctionCall]] = None,
    function_call: Optional[FunctionCall | str] = None,
    **kwargs: Any,
) -> ChatCompletion:
    """Request chat completion with streaming from API with proper retry logic.
    Functions are definitions of functions that can be called in the chat.

    Args:
        messages: List of messages to send to the chat.
        model_name: Name of the model to use. Defaults to ChatConfig.global_openai_model.
        functions:
            A list of functions the model may generate JSON inputs for.
            name: str / Required
                The name of the function to be called. Must be a-z, A-Z, 0-9,
                or contain underscores and dashes, with a maximum length of 64.
            description: str / Optional
                The description of what the function does.
            parameters: OpenAIFunctionParameter / Optional
                The parameters the functions accepts.
        function_call:
            Controls how the model responds to function calls. "none" means the model does not call a function,
            and responds to the end-user. "auto" means the model can pick between an end-user or calling a function.
            Specifying a particular function via {"name": "my_function"} forces the model to call that function.
            "none" is the default when no functions are present. "auto" is the default if functions are present.
    """
    kwargs.pop("stream", None)

    async def get_chat_completion(*args, **kwargs) -> ChatCompletion:
        async for chat_completion in acreate_chat_completion(*args, **kwargs):
            if TYPE_CHECKING:
                assert isinstance(chat_completion, ChatCompletion)
            return chat_completion
        raise ChatConnectionException(msg="Completion API error")

    return await acompletion_with_retry(
        get_chat_completion,
        messages=messages,
        model=model,
        api_base=api_base,
        api_key=api_key,
        functions=functions,
        function_call=function_call,
        stream=False,
        **kwargs,
    )


async def request_chat_completion_with_streaming(
    messages: List[Dict[str, str]],
    model: str = ChatConfig.global_openai_model,
    api_base: str = "https://api.openai.com/v1",
    api_key: Optional[str] = None,
    functions: Optional[List[FunctionCall]] = None,
    function_call: Optional[FunctionCall | str] = None,
    **kwargs: Any,
) -> AsyncIterator[ChatCompletionChunk]:
    """Request chat completion with streaming from API with proper retry logic.
    Functions are definitions of functions that can be called in the chat.

    Args:
        messages: List of messages to send to the chat.
        model_name: Name of the model to use. Defaults to ChatConfig.global_openai_model.
        functions:
            A list of functions the model may generate JSON inputs for.
            name: str / Required
                The name of the function to be called. Must be a-z, A-Z, 0-9,
                or contain underscores and dashes, with a maximum length of 64.
            description: str / Optional
                The description of what the function does.
            parameters: OpenAIFunctionParameter / Optional
                The parameters the functions accepts.
        function_call:
            Controls how the model responds to function calls. "none" means the model does not call a function,
            and responds to the end-user. "auto" means the model can pick between an end-user or calling a function.
            Specifying a particular function via {"name": "my_function"} forces the model to call that function.
            "none" is the default when no functions are present. "auto" is the default if functions are present.
    """
    kwargs.pop("stream", None)
    print(
        f"- DEBUG: ||request_chat_completion_with_streaming|| Sending messages: {messages}"
    )

    async def get_chat_completion_chunks(
        *args, **kwargs
    ) -> AsyncIterator[ChatCompletion | ChatCompletionChunk]:
        return acreate_chat_completion(*args, **kwargs)

    async for chunk in await acompletion_with_retry(
        get_chat_completion_chunks,
        model=model,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        functions=functions,
        function_call=function_call,
        stream=True,
        **kwargs,
    ):
        if TYPE_CHECKING:
            assert isinstance(chunk, ChatCompletionChunk)
        yield chunk


async def acreate_completion(
    prompt: str,
    model: str = ChatConfig.global_openai_model,
    api_base: str = "https://api.openai.com/v1/",
    api_key: Optional[str] = None,
    stream: bool = False,
    **kwargs: Any,
) -> AsyncIterator[Completion | CompletionChunk]:
    """Create Text Completion Iterator"""
    kwargs.pop("functions", None)
    kwargs.pop("function_call", None)

    url = api_base.rstrip("/") + "/completions"
    headers = _make_headers(api_key=api_key)
    data = orjson_dumps(
        kwargs | {"stream": stream, "model": model, "prompt": prompt}
    )
    async with ClientSession(
        timeout=ChatConfig.timeout
    ) as session:  # initialize client
        async with session.post(
            url,
            headers=headers,
            data=data,
        ) as response:
            await _handle_error_response(response)
            if stream:
                async for json_data in _extract_json_from_streaming_response(
                    response
                ):
                    yield make_completion_chunk_from_json(json_data)
            else:
                yield make_completion_from_json(await response.json())


async def acreate_chat_completion(
    messages: List[Dict[str, str]],
    model: str = ChatConfig.global_openai_model,
    api_base: str = "https://api.openai.com/v1",
    api_key: Optional[str] = None,
    functions: Optional[List[FunctionCall]] = None,
    function_call: Optional[FunctionCall | str] = None,
    stream: bool = False,
    **kwargs: Any,
) -> AsyncIterator[ChatCompletion | ChatCompletionChunk]:
    """Create Chat Completion Iterator"""
    url = api_base.rstrip("/") + "/chat/completions"
    headers = _make_headers(api_key=api_key)
    data = orjson_dumps(
        kwargs
        | {"stream": stream, "model": model, "messages": messages}
        | _decode_function_call(
            functions=functions, function_call=function_call
        )
    )
    async with ClientSession(
        timeout=ChatConfig.timeout,
    ) as session:  # initialize client
        async with session.post(
            url,
            headers=headers,
            data=data,
        ) as response:
            await _handle_error_response(response)
            if stream:
                async for json_data in _extract_json_from_streaming_response(
                    response
                ):
                    yield make_chat_completion_chunk_from_json(json_data)
            else:
                yield make_chat_completion_from_json(await response.json())
