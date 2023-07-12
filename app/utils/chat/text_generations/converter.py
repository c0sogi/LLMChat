from time import time
from typing import TYPE_CHECKING, Literal, Optional
from uuid import uuid4

from app.models.completion_models import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionLogprobs,
    CompletionUsage,
    FunctionCallUnparsed,
)

# ==== CHAT COMPLETION ====#


def make_chat_completion(
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    index: int = 0,
    id: Optional[str] = None,
    role: Optional[Literal["user", "system", "assistant"]] = None,
    created: Optional[int] = None,
    finish_reason: Optional[str] = None,
    user: Optional[str] = None,
    function_name: Optional[str] = None,
    function_args: Optional[str] = None,
) -> ChatCompletion:
    """A helper method to make a chat completion."""
    if id is None:
        id = f"cmpl-{str(uuid4())}"
    if created is None:
        created = int(time())
    if role is None:
        role = "assistant"
    message = ChatCompletionMessage(role=role, content=content)
    if user is not None:
        message["user"] = user
    if function_name is not None:
        function_call = FunctionCallUnparsed(name=function_name)
        if function_args is not None:
            function_call["arguments"] = function_args
        message["function_call"] = function_call
    return ChatCompletion(
        id=id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=index,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def make_chat_completion_from_json(json_data: dict, index: int = 0) -> ChatCompletion:  # type: ignore
    """Make ChatCompletion from json data(dict)"""
    if TYPE_CHECKING:
        # A hacky way to make mypy happy
        json_data: ChatCompletion = json_data  # type: ignore
    usage = json_data.get("usage")
    if usage is None:
        usage = CompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
    function_call = json_data["choices"][index]["message"].get("function_call")
    if function_call:
        function_name = function_call.get("name")
        function_arguments = function_call.get("arguments")
    else:
        function_name = None
        function_arguments = None
    return make_chat_completion(
        model=json_data["model"],
        content=json_data["choices"][index]["message"]["content"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        index=index,
        id=json_data.get("id"),
        role=json_data["choices"][index]["message"].get("role"),
        user=json_data["choices"][index]["message"].get("user"),
        created=json_data.get("created"),
        finish_reason=json_data["choices"][index].get("finish_reason"),
        function_name=function_name,
        function_args=function_arguments,
    )


def make_chat_completion_chunk(
    id: str,
    model: str,
    created: Optional[int] = None,
    role: Optional[Literal["assistant"]] = None,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    function_name: Optional[str] = None,
    function_args: Optional[str] = None,
) -> ChatCompletionChunk:
    """A helper method to make a chat completion chunk."""
    if created is None:
        created = int(time())
    delta = ChatCompletionChunkDelta()
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if function_name is not None or function_args is not None:
        function_call = FunctionCallUnparsed()
        if function_name is not None:
            function_call["name"] = function_name
        if function_args is not None:
            function_call["arguments"] = function_args
        delta["function_call"] = function_call
    return ChatCompletionChunk(
        id=id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
        ],
    )


def make_chat_completion_chunk_from_json(json_data: dict) -> ChatCompletionChunk:  # type: ignore
    """Make ChatCompletionChunk from json data(dict)"""
    if TYPE_CHECKING:
        # A hacky way to make mypy happy
        json_data: ChatCompletionChunk = json_data  # type: ignore
    delta = json_data["choices"][0]["delta"]
    function_call = delta.get("function_call")
    if function_call:
        function_name = function_call.get("name")
        function_arguments = function_call.get("arguments")
    else:
        function_name = None
        function_arguments = None
    return make_chat_completion_chunk(
        id=json_data["id"],
        model=json_data["model"],
        role=delta.get("role"),
        content=delta.get("content"),
        finish_reason=json_data["choices"][0].get("finish_reason"),
        function_name=function_name,
        function_args=function_arguments,
    )


# ==== TEXT COMPLETION ==== #


def make_completion_chunk(
    id: str,
    model: str,
    text: str,
    index: int = 0,
    finish_reason: Optional[str] = None,
    logprobs: Optional[CompletionLogprobs] = None,
    created: Optional[int] = None,
) -> CompletionChunk:
    """A helper method to make a completion chunk."""
    if created is None:
        created = int(time())
    return CompletionChunk(
        id=id,
        object="text_completion",
        created=created,
        model=model,
        choices=[
            CompletionChoice(
                text=text,
                index=index,
                logprobs=logprobs,
                finish_reason=finish_reason,
            )
        ],
    )


def make_completion_chunk_from_json(json_data: dict) -> CompletionChunk:  # type: ignore
    """Make CompletionChunk from json data(dict)"""
    if TYPE_CHECKING:
        # A hacky way to make mypy happy
        json_data: CompletionChunk = json_data  # type: ignore
    choice = json_data["choices"][0]
    return make_completion_chunk(
        id=json_data["id"],
        model=json_data["model"],
        text=choice["text"],
        index=choice.get("index", 0),
        finish_reason=choice.get("finish_reason"),
        logprobs=choice.get("logprobs"),
        created=json_data.get("created"),
    )


def make_completion(
    model: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    index: int = 0,
    id: Optional[str] = None,
    created: Optional[int] = None,
    finish_reason: Optional[str] = None,
    logprobs: Optional[CompletionLogprobs] = None,
) -> Completion:
    """A helper method to make a completion."""
    if id is None:
        id = f"cmst-{str(uuid4())}"
    if created is None:
        created = int(time())
    return Completion(
        id=id,
        object="text_completion",
        created=created,
        model=model,
        choices=[
            CompletionChoice(
                text=text,
                index=index,
                logprobs=logprobs,
                finish_reason=finish_reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def make_completion_from_json(json_data: dict, index: int = 0) -> Completion:  # type: ignore
    """Make Completion from json data(dict)"""
    if TYPE_CHECKING:
        # A hacky way to make mypy happy
        json_data: Completion = json_data  # type: ignore
    usage = json_data.get("usage")
    if usage is None:
        usage = CompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
    return make_completion(
        id=json_data["id"],
        model=json_data["model"],
        text=json_data["choices"][index]["text"],
        index=index,
        finish_reason=json_data["choices"][index].get("finish_reason"),
        logprobs=json_data["choices"][index].get("logprobs"),
        created=json_data.get("created"),
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    )
