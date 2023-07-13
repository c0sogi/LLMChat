"""A module for converting message histories to different formats."""

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from langchain import PromptTemplate

from app.common.constants import ChatTurnTemplates
from app.models.base_models import (
    APIChatMessage,
    MessageHistory,
    UserChatRoles,
)
from app.models.chat_models import ChatRoles

from .turn_templates import shatter_chat_turn_prompt

if TYPE_CHECKING:
    from langchain.schema import BaseMessage

T = TypeVar("T")


def langchain_parse_method(message_history: MessageHistory) -> "BaseMessage":
    """Parse message history to langchain message format."""
    from langchain.schema import (
        AIMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
    )

    if message_history.summarized is not None:
        message_history = deepcopy(message_history)
        if message_history.summarized is not None:
            message_history.content = message_history.summarized
    if message_history.actual_role == ChatRoles.USER.value:
        return HumanMessage(content=message_history.content)
    elif message_history.actual_role == ChatRoles.AI.value:
        return AIMessage(content=message_history.content)
    else:
        if message_history.role.startswith("function:"):
            return FunctionMessage(
                name=message_history.role.removeprefix("function:"),
                content=message_history.content,
            )
        return SystemMessage(content=message_history.content)


def chat_completion_api_parse_method(
    message_history: MessageHistory,
) -> dict[str, str]:
    """Parse message history to Chat Completion API message format.
    Used when sending message to Chat Completion API."""

    if message_history.summarized is not None:
        message_history = deepcopy(message_history)
        if message_history.summarized is not None:
            message_history.content = message_history.summarized
    if message_history.actual_role == ChatRoles.USER.value:
        return APIChatMessage(
            role="user",
            content=message_history.content,
        ).dict(exclude_none=True)
    elif message_history.actual_role == ChatRoles.AI.value:
        return APIChatMessage(
            role="assistant",
            content=message_history.content,
        ).dict(exclude_none=True)
    else:
        if message_history.role.startswith("function:"):
            return APIChatMessage(
                role="function",
                content=message_history.content,
                name=message_history.role.removeprefix("function:").strip(),
            ).dict(exclude_none=True)
        return APIChatMessage(
            role="system",
            content=message_history.content,
        ).dict(exclude_none=True)


def text_completion_api_parse_method(
    message_history: MessageHistory, chat_turn_prompt: PromptTemplate
) -> str:
    """Parse message history to Text Completion API message format.
    Used when sending message to Text Completion API."""

    if message_history.summarized is not None:
        message_history = deepcopy(message_history)
        if message_history.summarized is not None:
            message_history.content = message_history.summarized
    return chat_turn_prompt.format(
        role=message_history.role,
        content=message_history.content.strip(),
    )


# Frontend message format:
# message: msg["content"] ?? "",
# isGptSpeaking: msg["actual_role"] != "user" ? true : false,
# isFinished: true,
# datetime: parseLocaltimeFromTimestamp(msg["timestamp"]),
# modelName: msg["model_name"],
# uuid: msg["uuid"],


def init_parse_method(message_history: MessageHistory) -> dict[str, Any]:
    """Parse initial message history to frontend message format.
    Used when sending message to Flutter frontend."""

    return MessageHistory.from_orm(message_history).dict(
        exclude={
            "summarized",
            "summarized_tokens",
            "role",
            "tokens",
        }
    )


def message_histories_to_list(
    parse_method: Callable[[MessageHistory], T],
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: Optional[list[MessageHistory]] = None,
) -> list[T]:
    """Convert message histories to list of messages.
    Messages are sorted by timestamp.
    Prefix and suffix prompts are added to the list of messages."""

    return [
        parse_method(message_history)
        for message_history in sorted(
            user_message_histories
            + ai_message_histories
            + (system_message_histories or []),
            key=lambda m: m.timestamp,
        )
    ]


def message_histories_to_str(
    user_chat_roles: UserChatRoles,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: Optional[list[MessageHistory]] = None,
    parse_method: Optional[Callable[[MessageHistory], str]] = None,
    chat_turn_prompt: PromptTemplate = ChatTurnTemplates.ROLE_CONTENT_1,
) -> str:
    """Convert message histories to string.
    Messages are sorted by timestamp.
    Prefix and suffix prompts are added to the list of messages."""

    shattered: tuple[str, ...] = shatter_chat_turn_prompt(
        "role", "content", chat_turn_prompt=chat_turn_prompt
    )
    if parse_method is None:
        parse_method = partial(
            text_completion_api_parse_method,
            chat_turn_prompt=chat_turn_prompt,
        )

    return (
        "".join(
            message_histories_to_list(
                parse_method=parse_method,
                user_message_histories=user_message_histories,
                ai_message_histories=ai_message_histories,
                system_message_histories=system_message_histories,
            )
        )
        + f"{shattered[0]}{user_chat_roles.ai}{shattered[2]}"
    )
