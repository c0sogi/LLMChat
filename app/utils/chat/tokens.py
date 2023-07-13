from bisect import bisect_left
from typing import TYPE_CHECKING, Optional, Union

from langchain import PromptTemplate

from app.common.config import ChatConfig
from app.models.chat_models import ChatRoles, MessageHistory, UserChatContext

if TYPE_CHECKING:
    from app.models.llms import LLMModel


def get_token_limit_with_n_messages(
    user_chat_context: UserChatContext,
    n_user_messages: int,
    n_ai_messages: int,
    n_system_messages: int,
    prefix_prompt_tokens: int = 0,
    suffix_prompt_tokens: int = 0,
) -> int:
    """
    Get the number of tokens left in the LLM model,
    with the given number of messages from each message history.
    This is used to determine if the LLM model has enough tokens to generate a response.
    """
    llm_model: LLMModel = user_chat_context.llm_model.value
    users: list[MessageHistory] = user_chat_context.user_message_histories
    ais: list[MessageHistory] = user_chat_context.ai_message_histories
    syss: list[MessageHistory] = user_chat_context.system_message_histories

    return llm_model.max_total_tokens - (
        sum([m.tokens for m in users[-min(n_user_messages, len(users)) :]])
        + sum([m.tokens for m in ais[-min(n_ai_messages, len(ais)) :]])
        + sum([m.tokens for m in syss[-min(n_system_messages, len(syss)) :]])
        + prefix_prompt_tokens
        + suffix_prompt_tokens
        + llm_model.token_margin
        + ChatConfig.extra_token_margin
    )


def make_formatted_query(
    user_chat_context: UserChatContext,
    question: str,
    context: str,
    query_template: Union[PromptTemplate, str],
    with_n_user_messages: int = 0,
    with_n_ai_messages: int = 0,
    with_n_system_messages: int = 0,
) -> str:
    """Make a formatted query to the LLM model, with the given question and context.
    Token limit is calculated based on the number of messages in the user, AI, and system message histories.
    """
    llm_model = user_chat_context.llm_model.value
    token_limit: int = (
        get_token_limit_with_n_messages(
            user_chat_context=user_chat_context,
            n_user_messages=with_n_user_messages,
            n_ai_messages=with_n_ai_messages,
            n_system_messages=with_n_system_messages,
            suffix_prompt_tokens=llm_model.suffix_tokens,
            prefix_prompt_tokens=llm_model.prefix_tokens,
        )
        - 100
    )
    context = llm_model.tokenizer.get_chunk_of(
        context,
        tokens=token_limit
        - user_chat_context.get_tokens_of(
            query_template.format(context="", question=question)
        ),
    )
    return query_template.format(context=context, question=question)


def make_truncated_text(
    user_chat_context: UserChatContext,
    text: str,
    with_n_user_messages: int = 0,
    with_n_ai_messages: int = 0,
    with_n_system_messages: int = 0,
) -> str:
    llm_model = user_chat_context.llm_model.value
    token_limit: int = (
        get_token_limit_with_n_messages(
            user_chat_context=user_chat_context,
            n_user_messages=with_n_system_messages,
            n_ai_messages=with_n_user_messages,
            n_system_messages=with_n_ai_messages,
            suffix_prompt_tokens=llm_model.suffix_tokens,
            prefix_prompt_tokens=llm_model.prefix_tokens,
        )
        - 100
    )
    return llm_model.tokenizer.get_chunk_of(text, tokens=token_limit)


def cutoff_message_histories(
    user_chat_context: UserChatContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    token_limit: int,
) -> tuple[list[MessageHistory], list[MessageHistory], list[MessageHistory]]:
    """
    Cutoff message histories to fit the token limit.
    Forget the oldest messages when token limit is exceeded
    """
    # Separate the prefix and suffix messages, and precompute the number of tokens.
    prefix_message: Optional[MessageHistory] = None
    suffix_message: Optional[MessageHistory] = None
    llm_model: "LLMModel" = user_chat_context.llm_model.value
    prefix_prompt = llm_model.prefix
    suffix_prompt = llm_model.suffix

    if prefix_prompt:
        prefix_message = MessageHistory(
            content=prefix_prompt,
            tokens=llm_model.prefix_tokens,
            role=llm_model.user_chat_roles.system,
            actual_role=ChatRoles.SYSTEM.value,
            timestamp=-1,  # This is a dummy timestamp.
        )
    if suffix_prompt:
        suffix_message = MessageHistory(
            content=suffix_prompt,
            tokens=llm_model.suffix_tokens,
            role=llm_model.user_chat_roles.system,
            actual_role=ChatRoles.SYSTEM.value,
            timestamp=2**50,  # This is a dummy timestamp.
        )

    # Calculates a cap on the number of tokens excluding prefix and suffix messages.
    remaining_tokens = (
        token_limit - prefix_message.tokens
        if prefix_message
        else token_limit - suffix_message.tokens
        if suffix_message
        else token_limit
    )
    print(f"- DEBUG: remaining_tokens: {remaining_tokens}", flush=True)

    # If the remaining tokens are negative, return an empty tuple.
    if remaining_tokens < 0:
        return ([], [], [])

    # Sort all messages by timestamp and filter out the prefix and suffix messages.
    messages_without_prefix_and_suffix: list[MessageHistory] = sorted(
        user_message_histories
        + ai_message_histories
        + [m for m in system_message_histories if not m.is_prefix and not m.is_suffix],
        key=lambda m: m.timestamp,
    )

    # If the total tokens of all messages are less than or equal to the remaining tokens, return the input as it is.
    if sum(m.tokens for m in messages_without_prefix_and_suffix) <= remaining_tokens:
        _system_message_histories = [
            m
            for m in messages_without_prefix_and_suffix
            if m in system_message_histories
        ]
        if prefix_message:
            _system_message_histories.insert(0, prefix_message)
        if suffix_message:
            _system_message_histories.append(suffix_message)
        return (
            user_message_histories,
            ai_message_histories,
            _system_message_histories,
        )

    # Get the cumulative tokens of all messages.
    all_tokens = [0]
    for m in messages_without_prefix_and_suffix:
        all_tokens.append(all_tokens[-1] + m.tokens)

    # Find the index of the first message that fits the remaining tokens using binary search.
    index = bisect_left(all_tokens, all_tokens[-1] - remaining_tokens)

    # Slice the selected messages from the index to the end.
    selected_messages: list[MessageHistory] = messages_without_prefix_and_suffix[index:]

    # Separate selected messages by each type using list comprehensions.
    user_messages = [m for m in selected_messages if m in user_message_histories]
    ai_messages = [m for m in selected_messages if m in ai_message_histories]

    # Add prefix and suffix messages to the system message.
    system_messages = [m for m in selected_messages if m in system_message_histories]
    if prefix_message:
        system_messages.insert(0, prefix_message)
    if suffix_message:
        system_messages.append(suffix_message)
    # Returns a list of messages for each type as a tuple.
    return (user_messages, ai_messages, system_messages)
