import asyncio
from collections import deque
from copy import deepcopy
from typing import Any, Optional

from langchain import PromptTemplate
from orjson import loads as orjson_loads

from app.common.config import ChatConfig
from app.common.constants import JSON_PATTERN
from app.models.chat_models import ChatRoles, MessageHistory
from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext

from app.utils.chat.messages.converter import (
    message_histories_to_list,
    openai_parse_method,
)
from app.utils.chat.tokens import cutoff_message_histories
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.langchain.chat_openai import CustomChatOpenAI

shared = Shared()


async def aget_query_to_search(
    buffer: BufferedUserContext,
    query: str,
    search_llm: CustomChatOpenAI,
) -> str:
    user_message_histories, ai_message_histories = cutoff_message_histories(
        ai_message_histories=buffer.current_ai_message_histories,
        user_message_histories=buffer.current_user_message_histories
        + [
            MessageHistory(
                role=buffer.current_user_chat_roles.user,
                content=query,
                tokens=buffer.current_user_chat_context.get_tokens_of(query)
                + buffer.current_llm_model.value.token_margin,
                actual_role=ChatRoles.USER.value,
            )
        ],
        system_message_histories=[],
        token_limit=ChatConfig.query_context_token_limit,
    )
    web_search_llm_output = (
        await search_llm.agenerate(
            messages=[
                message_histories_to_list(
                    user_chat_roles=buffer.current_user_chat_roles,
                    parse_method=openai_parse_method,
                    user_message_histories=user_message_histories,
                    ai_message_histories=ai_message_histories,
                )
            ],
        )
    ).llm_output
    if web_search_llm_output is None:
        return query
    query_to_search = web_search_llm_output["function_calls"][0]["arguments"][
        "query_to_search"
    ]
    await SendToWebsocket.message(
        websocket=buffer.websocket,
        msg="---\n{query_to_search}\n```\n".format(
            query_to_search=query_to_search.replace("```", "'''")
        ),
        chat_room_id=buffer.current_chat_room_id,
        finish=False,
    )
    return query_to_search


async def aget_json(
    query_template: PromptTemplate, **kwargs_to_format: str
) -> Optional[Any]:
    try:
        json_query = JSON_PATTERN.search(
            await Shared().llm.apredict(  # type: ignore
                query_template.format(**kwargs_to_format)
            )
        )
        if json_query is None:
            raise ValueError("Result is None")
        else:
            return orjson_loads(json_query.group())
    except Exception:
        return
