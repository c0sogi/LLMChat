from typing import Any, AsyncGenerator

import asyncio
from langchain.callbacks import AsyncIteratorCallbackHandler
from app.errors.chat_exceptions import ChatTextGenerationException
from openai.error import OpenAIError
from app.models.chat_models import MessageHistory

from app.models.llms import OpenAIModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.messages.converter import (
    message_histories_to_list,
    openai_parse_method,
)
from app.utils.langchain.chat_openai import CustomChatOpenAI

CUSTOM_CHAT_OPENAI: "CustomChatOpenAI" = CustomChatOpenAI(client=None, streaming=True)


async def agenerate_from_openai(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    current_model = buffer.current_llm_model.value
    assert isinstance(current_model, OpenAIModel)

    user_defined_api_key: str | None = (
        buffer.current_user_chat_context.optional_info.get("api_key")
    )
    api_key_to_use: Any = (
        user_defined_api_key
        if user_defined_api_key is not None
        else current_model.api_key
    )

    callback = AsyncIteratorCallbackHandler()
    producer = CUSTOM_CHAT_OPENAI.agenerate(
        stop=None,
        messages=[
            message_histories_to_list(
                user_chat_roles=buffer.current_user_chat_roles,
                parse_method=openai_parse_method,
                user_message_histories=user_message_histories,
                ai_message_histories=ai_message_histories,
                system_message_histories=system_message_histories,
                prefix_prompt=current_model.prefix,
                prefix_prompt_tokens=current_model.prefix_tokens,
                suffix_prompt=current_model.suffix,
                suffix_prompt_tokens=current_model.suffix_tokens,
            ),
        ],
        callbacks=[callback],
        temperature=buffer.current_user_chat_profile.temperature,
        top_p=buffer.current_user_chat_profile.top_p,
        presence_penalty=buffer.current_user_chat_profile.presence_penalty,
        frequency_penalty=buffer.current_user_chat_profile.frequency_penalty,
        logit_bias={},
        user=buffer.user_id,
        max_tokens=max_tokens,
        model=current_model.name,
        api_key=api_key_to_use,
    )

    producer_task = asyncio.create_task(producer)

    try:
        async for token in callback.aiter():
            if buffer.done.is_set():
                raise InterruptedError("Chat was interrupted by the user.")
            yield token
        await producer_task

    finally:
        if not producer_task.done():
            producer_task.cancel()
        try:
            await producer_task
        except OpenAIError as e:
            raise ChatTextGenerationException(msg=str(e))
        except asyncio.CancelledError:
            pass
