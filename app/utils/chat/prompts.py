# flake8: noqa

from copy import deepcopy
from itertools import zip_longest
from typing import Any, Callable, Optional

from app.errors.chat_exceptions import ChatTooMuchTokenException
from app.models.chat_models import MessageHistory
from app.viewmodels.base_models import UserChatRoles


def message_histories_to_list(
    parse_method: Callable[[MessageHistory], Any],
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: Optional[list[MessageHistory]] = None,
) -> list[Any]:
    message_histories: list[dict[str, str]] = []
    if system_message_histories:
        for system_history in system_message_histories:
            message_histories.append(parse_method(system_history))  # append system message history
    for user_message_history, ai_message_history in zip_longest(
        user_message_histories,
        ai_message_histories,
    ):
        if user_message_history is not None:
            message_histories.append(parse_method(user_message_history))
        if ai_message_history is not None:
            message_histories.append(parse_method(ai_message_history))
    return message_histories


def message_histories_to_str(
    user_chat_roles: UserChatRoles,
    chat_turn_prompt: str,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: Optional[list[MessageHistory]] = None,
    description_for_prompt: Optional[str] = None,
):
    def parse_method(message_history: MessageHistory) -> str:
        if message_history.summarized is not None:
            message_history = deepcopy(message_history)
            if message_history.summarized is not None:
                message_history.content = message_history.summarized
        return chat_turn_prompt.format(role=message_history.role, content=message_history.content.strip())

    if description_for_prompt:
        prefix: str = description_for_prompt.format(  # type: ignore
            user=user_chat_roles.user,
            USER=user_chat_roles.user,
            ai=user_chat_roles.ai,
            AI=user_chat_roles.ai,
            system=user_chat_roles.system,
            SYSTEM=user_chat_roles.system,
        )
    else:
        prefix: str = ""

    return (
        prefix
        + "".join(
            message_histories_to_list(
                parse_method=parse_method,
                user_message_histories=user_message_histories,
                ai_message_histories=ai_message_histories,
                system_message_histories=system_message_histories,
            )
        )
        + chat_turn_prompt.format(role=user_chat_roles.ai, content="").strip()
        + " "
    )


def cutoff_message_histories(
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    token_limit: int,
    extra_token_margin: Optional[int] = None,
) -> tuple[list[MessageHistory], list[MessageHistory]]:
    system_message_tokens: int = sum(
        [system_message_history.tokens for system_message_history in system_message_histories]
    )
    if system_message_tokens > token_limit:
        raise ChatTooMuchTokenException(msg="System messages exceed the token limit.")

    token_limit -= system_message_tokens
    user_results: list[MessageHistory] = []
    ai_results: list[MessageHistory] = []

    num_user_messages = len(user_message_histories)
    num_ai_messages = len(ai_message_histories)
    idx = 1

    while (num_user_messages - idx >= 0 or num_ai_messages - idx >= 0) and token_limit > 0:
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
