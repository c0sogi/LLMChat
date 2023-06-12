# flake8: noqa
from copy import deepcopy
from itertools import zip_longest
from re import sub
from functools import partial
from typing import Any, Callable, Optional

from langchain import PromptTemplate

from app.common.constants import ChatTurnTemplates
from app.errors.chat_exceptions import ChatTooMuchTokenException
from app.models.chat_models import MessageHistory
from app.models.base_models import (
    OpenAIChatMessage,
    SendInitToWebsocket,
    UserChatRoles,
)


def str_parse_method(
    message_history: MessageHistory, chat_turn_prompt: PromptTemplate
) -> str:
    if message_history.summarized is not None:
        message_history = deepcopy(message_history)
        if message_history.summarized is not None:
            message_history.content = message_history.summarized
    return chat_turn_prompt.format(
        role=message_history.role, content=message_history.content.strip()
    )


def openai_parse_method(message_history: MessageHistory) -> dict[str, str]:
    if message_history.summarized is not None:
        message_history = deepcopy(message_history)
        if message_history.summarized is not None:
            message_history.content = message_history.summarized
    if message_history.is_user:
        message_history.role = "user"
    else:
        message_history.role = "assistant"
    return OpenAIChatMessage.from_orm(message_history).dict()


def init_parse_method(message_history: MessageHistory) -> dict[str, Any]:
    return SendInitToWebsocket.from_orm(message_history).dict()


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
                    is_user=False,
                )
            )
        )
    if system_message_histories:
        for system_history in system_message_histories:
            message_histories.append(
                parse_method(system_history)
            )  # append system message history
    for user_message_history, ai_message_history in zip_longest(
        user_message_histories,
        ai_message_histories,
    ):
        if user_message_history is not None:
            message_histories.append(parse_method(user_message_history))
        if ai_message_history is not None:
            message_histories.append(parse_method(ai_message_history))
    if suffix_prompt is not None:
        message_histories.append(
            parse_method(
                MessageHistory(
                    role=user_chat_roles.system,
                    content=suffix_prompt,
                    tokens=suffix_prompt_tokens,
                    is_user=False,
                )
            )
        )
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
):
    return "".join(
        message_histories_to_list(
            user_chat_roles=user_chat_roles,
            parse_method=partial(str_parse_method, chat_turn_prompt=chat_turn_prompt)
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
    ) + chat_turn_prompt.format(role=user_chat_roles.ai, content="")


def cutoff_message_histories(
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    token_limit: int,
    extra_token_margin: Optional[int] = None,
) -> tuple[list[MessageHistory], list[MessageHistory]]:
    system_message_tokens: int = sum(
        [
            system_message_history.tokens
            for system_message_history in system_message_histories
        ]
    )
    if system_message_tokens > token_limit:
        raise ChatTooMuchTokenException(msg="System messages exceed the token limit.")

    token_limit -= system_message_tokens
    user_results: list[MessageHistory] = []
    ai_results: list[MessageHistory] = []

    num_user_messages = len(user_message_histories)
    num_ai_messages = len(ai_message_histories)
    idx = 1

    while (
        num_user_messages - idx >= 0 or num_ai_messages - idx >= 0
    ) and token_limit > 0:
        user_and_ai_tokens = 0
        if num_user_messages - idx >= 0:
            user_and_ai_tokens += user_message_histories[-idx].tokens
        if num_ai_messages - idx >= 0:
            user_and_ai_tokens += ai_message_histories[-idx].tokens

        if user_and_ai_tokens <= token_limit:
            if num_user_messages - idx >= 0:
                user_results.append(user_message_histories[-idx])
            if num_ai_messages - idx >= 0:
                ai_results.append(ai_message_histories[-idx])
            token_limit -= user_and_ai_tokens
        else:
            break
        idx += 1

    if extra_token_margin is not None:
        deleted_tokens: int = 0
        while user_results and ai_results and deleted_tokens < extra_token_margin:
            deleted_tokens += user_results.pop().tokens + ai_results.pop().tokens

    return list(reversed(user_results)), list(reversed(ai_results))
