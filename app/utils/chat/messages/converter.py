from copy import deepcopy
from functools import partial
from typing import Any, Callable, Optional

from langchain import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

from app.common.constants import ChatTurnTemplates
from app.models.chat_models import ChatRoles, MessageHistory
from app.models.base_models import (
    APIChatMessage,
    MessageHistory,
    UserChatRoles,
)
from app.utils.logger import ApiLogger


def openai_parse_method(message_history: MessageHistory) -> BaseMessage:
    if message_history.summarized is not None:
        message_history = deepcopy(message_history)
        if message_history.summarized is not None:
            message_history.content = message_history.summarized
    if message_history.actual_role == ChatRoles.USER.value:
        return HumanMessage(content=message_history.content)
    elif message_history.actual_role == ChatRoles.AI.value:
        return AIMessage(content=message_history.content)
    else:
        return SystemMessage(content=message_history.content)


def chat_completion_api_parse_method(message_history: MessageHistory) -> dict:
    if message_history.summarized is not None:
        message_history = deepcopy(message_history)
        if message_history.summarized is not None:
            message_history.content = message_history.summarized
    if message_history.actual_role == ChatRoles.USER.value:
        return APIChatMessage(
            role="user",
            content=message_history.content,
        ).dict()
    elif message_history.actual_role == ChatRoles.AI.value:
        return APIChatMessage(
            role="assistant",
            content=message_history.content,
        ).dict()
    else:
        return APIChatMessage(
            role="system",
            content=message_history.content,
        ).dict()


def text_completion_api_parse_method(
    message_history: MessageHistory, chat_turn_prompt: PromptTemplate
) -> str:
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
    return MessageHistory.from_orm(message_history).dict(
        exclude={
            "summarized",
            "summarized_tokens",
            "role",
            "tokens",
        }
    )


def message_histories_to_list(
    user_chat_roles: UserChatRoles,
    parse_method: Callable[[MessageHistory], Any],
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: Optional[list[MessageHistory]] = None,
    prefix_prompt: Optional[str] = None,
    prefix_prompt_tokens: int = 0,
    suffix_prompt: Optional[str] = None,
    suffix_prompt_tokens: int = 0,
) -> list[Any]:
    message_histories: list[Any] = []
    if prefix_prompt is not None:
        message_histories.append(
            parse_method(
                MessageHistory(
                    role=user_chat_roles.system,
                    content=prefix_prompt,
                    tokens=prefix_prompt_tokens,
                    actual_role=ChatRoles.SYSTEM.value,
                )
            )
        )

    message_histories.extend(
        [
            parse_method(message_history)
            for message_history in sorted(
                user_message_histories
                + ai_message_histories
                + (system_message_histories if system_message_histories else []),
                key=lambda x: x.timestamp,
            )
        ]
    )  # organize message histories
    if suffix_prompt is not None:
        message_histories.append(
            parse_method(
                MessageHistory(
                    role=user_chat_roles.system,
                    content=suffix_prompt,
                    tokens=suffix_prompt_tokens,
                    actual_role=ChatRoles.SYSTEM.value,
                )
            )
        )
    # ApiLogger("||message_histories_to_list||").info(
    #     f"Sending these messages to LLM:\n{message_histories}"
    # )
    return message_histories


def message_histories_to_str(
    user_chat_roles: UserChatRoles,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: Optional[list[MessageHistory]] = None,
    prefix_prompt: Optional[str] = None,
    prefix_prompt_tokens: int = 0,
    suffix_prompt: Optional[str] = None,
    suffix_prompt_tokens: int = 0,
    parse_method: Optional[Callable[[MessageHistory], Any]] = None,
    chat_turn_prompt: PromptTemplate = ChatTurnTemplates.ROLE_CONTENT_1,
    unique_str: str = "𐂂B㍱༒௵𒉓𒉓𒉓꩜",
):
    suffix = chat_turn_prompt.format(role=user_chat_roles.ai, content=unique_str)
    return (
        "".join(
            message_histories_to_list(
                user_chat_roles=user_chat_roles,
                parse_method=partial(
                    text_completion_api_parse_method,
                    chat_turn_prompt=chat_turn_prompt,
                )
                if parse_method is None
                else parse_method,
                user_message_histories=user_message_histories,
                ai_message_histories=ai_message_histories,
                system_message_histories=system_message_histories,
                prefix_prompt=prefix_prompt,
                prefix_prompt_tokens=prefix_prompt_tokens,
                suffix_prompt=suffix_prompt,
                suffix_prompt_tokens=suffix_prompt_tokens,
            )
        )
        + suffix[: suffix.find(unique_str)]
    )
