from typing import AsyncIterator, Literal, Optional
from urllib.parse import urlparse

from langchain import PromptTemplate

from app.common.config import config
from app.common.constants import ChatTurnTemplates
from app.errors.chat_exceptions import (
    ChatContentFilterException,
    ChatFunctionCallException,
    ChatLengthException,
)
from app.models.base_models import MessageHistory
from app.models.completion_models import FunctionCallUnparsed
from app.models.function_calling.base import FunctionCall
from app.models.llms import ExllamaModel, LlamaCppModel, OpenAIModel
from app.utils.api.completion import (
    request_chat_completion_with_streaming,
    request_text_completion_with_streaming,
)
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.messages.converter import (
    chat_completion_api_parse_method,
    message_histories_to_list,
    message_histories_to_str,
)
from app.utils.chat.messages.turn_templates import identify_end_of_string
from app.utils.function_calling.parser import (
    make_function_call_parsed_from_dict,
)
from app.utils.function_calling.token_count import (
    get_num_tokens_from_functions,
)


def _get_stop_strings(
    *roles: str, chat_turn_prompt: PromptTemplate
) -> list[str]:
    """Get stop strings for text completion API.
    Stop strings are required to stop text completion API from generating
    text that does not belong to the current chat turn.
    e.g. The common stop string is "### USER:", which can prevent ai from generating
    user's message itself."""

    prompt_stop = set()
    eos: Optional[str] = identify_end_of_string(
        "role", "content", chat_turn_prompt=chat_turn_prompt
    )
    if eos:
        prompt_stop.add(eos)
    for role in roles:
        avoids = (
            chat_turn_prompt.format(role=role, content="").strip(),
            f"{role}:",
            f"### {role}:",
            f"###{role}:",
        )
        prompt_stop.update(
            avoids,
            map(str.capitalize, avoids),
            map(str.upper, avoids),
            map(str.lower, avoids),
        )
    return list(prompt_stop)


def _get_api_key(buffer: BufferedUserContext) -> Optional[str]:
    """Return API key to use for completion API"""
    user_defined_api_key: Optional[str] = buffer.optional_info.get("api_key")
    return (
        str(user_defined_api_key)
        if user_defined_api_key is not None
        else getattr(buffer.current_llm_model.value, "api_key", None)
    )


def _get_api_base(buffer: BufferedUserContext) -> str:
    """Return API base to use for completion API"""
    current_model = buffer.current_llm_model.value
    if isinstance(current_model, (LlamaCppModel, ExllamaModel)):
        api_url = config.llama_completion_url
        assert api_url is not None
    else:
        api_url = current_model.api_url

    parsed_url = urlparse(api_url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}/v1"


def _make_common_kwargs(buffer: BufferedUserContext, max_tokens: int) -> dict:
    """Make common kwargs to use for completion API"""
    if isinstance(buffer.current_llm_model.value, OpenAIModel):
        model = buffer.current_llm_model.value.name
    else:
        model = buffer.current_llm_model.name
    return dict(
        model=model,
        temperature=buffer.current_user_chat_profile.temperature,
        top_p=buffer.current_user_chat_profile.top_p,
        presence_penalty=buffer.current_user_chat_profile.presence_penalty,
        frequency_penalty=buffer.current_user_chat_profile.frequency_penalty,
        user=buffer.user_id,
        max_tokens=max_tokens,
        api_key=_get_api_key(buffer=buffer),
        api_base=_get_api_base(buffer=buffer),
    )


def _make_chat_completion_kwargs(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> dict:
    """Make kwargs to use for chat completion API"""
    functions: Optional[list[FunctionCall]] = buffer.optional_info.get(
        "functions"
    )
    function_call: Optional[
        FunctionCall | Literal["auto", "none"]
    ] = buffer.optional_info.get("function_call")
    if functions is not None:
        max_tokens -= get_num_tokens_from_functions(functions)
    return _make_common_kwargs(buffer=buffer, max_tokens=max_tokens) | {
        "messages": message_histories_to_list(
            parse_method=chat_completion_api_parse_method,
            user_message_histories=user_message_histories,
            ai_message_histories=ai_message_histories,
            system_message_histories=system_message_histories,
        ),
        "functions": functions,
        "function_call": function_call,
    }


def _make_text_completion_kwargs(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
):
    chat_turn_prompt: PromptTemplate = (
        buffer.current_llm_model.value.chat_turn_prompt
        if isinstance(buffer.current_llm_model.value, LlamaCppModel)
        else ChatTurnTemplates.ROLE_CONTENT_1
    )
    return _make_common_kwargs(buffer=buffer, max_tokens=max_tokens) | {
        "prompt": message_histories_to_str(
            user_chat_roles=buffer.current_user_chat_roles,
            user_message_histories=user_message_histories,
            ai_message_histories=ai_message_histories,
            system_message_histories=system_message_histories,
            chat_turn_prompt=chat_turn_prompt,
        ),
        "stop": _get_stop_strings(
            buffer.current_llm_model.value.user_chat_roles.user,
            buffer.current_llm_model.value.user_chat_roles.ai,
            buffer.current_llm_model.value.user_chat_roles.system,
            chat_turn_prompt=chat_turn_prompt,
        ),
    }


async def agenerate_from_chat_completion_api(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> AsyncIterator[str]:
    content: str = ""
    function_call_name: str = ""
    function_call_arguments: str = ""
    async for chat_completion_chunk in request_chat_completion_with_streaming(
        **_make_chat_completion_kwargs(
            buffer=buffer,
            user_message_histories=user_message_histories,
            ai_message_histories=ai_message_histories,
            system_message_histories=system_message_histories,
            max_tokens=max_tokens,
        )
    ):
        _finish_reason = chat_completion_chunk["choices"][0]["finish_reason"]
        _content = chat_completion_chunk["choices"][0]["delta"].get("content")
        _function_call = chat_completion_chunk["choices"][0]["delta"].get(
            "function_call"
        )

        if _content:
            content += _content
            yield _content
        if _function_call is not None:
            function_call_name += str(_function_call.get("name", ""))
            function_call_arguments += str(_function_call.get("arguments", ""))
        if _finish_reason not in ("null", None):
            # End-of-Stream
            if _finish_reason == "length":
                raise ChatLengthException(
                    msg=content
                )  # raise exception for token limit
            elif _finish_reason == "content_filter":
                raise ChatContentFilterException(
                    msg="Omitted content due to a flag from our content filters"
                )  # raise exception for openai content filter
            elif function_call_name:
                # Raise exception for function call
                function_call_unparsed = FunctionCallUnparsed(
                    name=function_call_name
                )
                if function_call_arguments:
                    function_call_unparsed[
                        "arguments"
                    ] = function_call_arguments
                function_call_parsed = make_function_call_parsed_from_dict(
                    function_call_unparsed
                )
                raise ChatFunctionCallException(
                    func_name=function_call_parsed["name"],
                    func_kwargs=function_call_parsed.get("arguments", {}),
                )


async def agenerate_from_text_completion_api(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> AsyncIterator[str]:
    text: str = ""
    async for completion_chunk in request_text_completion_with_streaming(
        **_make_text_completion_kwargs(
            buffer=buffer,
            user_message_histories=user_message_histories,
            ai_message_histories=ai_message_histories,
            system_message_histories=system_message_histories,
            max_tokens=max_tokens,
        )
    ):
        _finish_reason = completion_chunk["choices"][0]["finish_reason"]
        _text = completion_chunk["choices"][0]["text"]

        if _text:
            text += _text
            yield _text

        if _finish_reason not in ("null", None):
            if _finish_reason == "length":
                raise ChatLengthException(
                    msg=text
                )  # raise exception for token limit
            elif _finish_reason == "content_filter":
                raise ChatContentFilterException(
                    msg="Omitted content due to a flag from our content filters"
                )  # raise exception for openai content filter
