from typing import Optional, Union

from langchain import PromptTemplate
from app.common.config import ChatConfig

from app.errors.chat_exceptions import ChatTextGenerationException
from app.models.chat_models import MessageHistory, UserChatContext
from app.models.llms import LLMModel
from app.utils.chat.buffer import BufferedUserContext


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
    llm_model = user_chat_context.llm_model.value
    user_tokens: int = sum(
        [
            m.tokens
            for m in user_chat_context.user_message_histories[
                -min(n_user_messages, len(user_chat_context.user_message_histories)) :
            ]
        ]
    )
    ai_tokens: int = sum(
        [
            m.tokens
            for m in user_chat_context.ai_message_histories[
                -min(n_ai_messages, len(user_chat_context.ai_message_histories)) :
            ]
        ]
    )
    system_tokens: int = sum(
        [
            m.tokens
            for m in user_chat_context.system_message_histories[
                -min(
                    n_system_messages, len(user_chat_context.system_message_histories)
                ) :
            ]
        ]
    )
    return llm_model.max_total_tokens - (
        user_tokens
        + ai_tokens
        + system_tokens
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
) -> str:
    """Make a formatted query to the LLM model, with the given question and context.
    Token limit is calculated based on the number of messages in the user, AI, and system message histories.
    """
    llm_model = user_chat_context.llm_model.value
    token_limit: int = get_token_limit_with_n_messages(
        user_chat_context=user_chat_context,
        n_ai_messages=0,
        n_system_messages=0,
        n_user_messages=0,
        suffix_prompt_tokens=llm_model.suffix_tokens,
        prefix_prompt_tokens=llm_model.prefix_tokens,
    )
    left_tokens: int = (
        token_limit
        - user_chat_context.get_tokens_of(
            query_template.format(context="", question=question)
        )
        - llm_model.token_margin
    )
    context = llm_model.tokenizer.split_text_on_tokens(
        context,
        tokens_per_chunk=left_tokens - llm_model.token_margin,
        chunk_overlap=0,
    )[0]
    return query_template.format(context=context, question=question)


def cutoff_message_histories(
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    token_limit: int,
    extra_token_margin: Optional[int] = None,
) -> tuple[list[MessageHistory], list[MessageHistory]]:
    """
    Cutoff message histories to fit the token limit.
    Forget the oldest messages when token limit is exceeded
    """
    system_message_tokens: int = sum(
        [
            system_message_history.tokens
            for system_message_history in system_message_histories
        ]
    )
    if system_message_tokens > token_limit:
        raise ChatTextGenerationException(msg="System messages exceed the token limit.")

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
