from asyncio import wait_for
from typing import Any, Literal, Optional

from app.common.config import OPENAI_API_KEY, ChatConfig
from app.models.completion_models import FunctionCallParsed
from app.models.function_calling.base import FunctionCall
from app.utils.api.completion import request_chat_completion

from .parser import make_function_call_parsed_from_dict


async def request_function_call(
    messages: list[dict[str, str]],
    functions: list[FunctionCall],
    function_call: Optional[FunctionCall | Literal["auto", "none"]] = "auto",
    model: str = ChatConfig.global_openai_model,
    api_base: str = "https://api.openai.com/v1",
    api_key: Optional[str] = OPENAI_API_KEY,
    timeout: Optional[float] = None,
    force_arguments: bool = False,
    **kwargs: Any,
) -> FunctionCallParsed:
    coro = request_chat_completion(
        messages=messages,
        model=model,
        api_base=api_base,
        api_key=api_key,
        functions=functions,
        function_call=function_call,
        **kwargs,
    )
    if timeout is not None:
        coro = wait_for(coro, timeout=timeout)
    function_call_unparsed = (await coro)["choices"][0]["message"].get(
        "function_call"
    )
    if function_call_unparsed is None:
        raise ValueError("No function call returned")
    function_call_parsed = make_function_call_parsed_from_dict(
        function_call_unparsed
    )
    if force_arguments and "arguments" not in function_call_parsed:
        raise ValueError("No arguments returned")

    return function_call_parsed
