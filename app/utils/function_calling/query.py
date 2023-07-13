from typing import Any, Optional

from langchain import PromptTemplate
from orjson import loads as orjson_loads

from app.common.config import ChatConfig
from app.common.constants import JSON_PATTERN
from app.models.chat_models import ChatRoles, MessageHistory
from app.models.function_calling.base import FunctionCall
from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.messages.converter import (
    chat_completion_api_parse_method,
    message_histories_to_list,
)
from app.utils.chat.tokens import cutoff_message_histories

from .request import request_function_call


async def aget_query_to_search(
    buffer: BufferedUserContext,
    query: str,
    function: FunctionCall,
    timeout: Optional[float] = 30,
) -> str:
    """Get query to search from user query and current context"""
    (
        user_message_histories,
        ai_message_histories,
        system_message_histories,
    ) = cutoff_message_histories(
        user_chat_context=buffer.current_user_chat_context,
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
    try:
        function_call_parsed = await request_function_call(
            messages=message_histories_to_list(
                parse_method=chat_completion_api_parse_method,
                user_message_histories=user_message_histories,
                ai_message_histories=ai_message_histories,
                system_message_histories=system_message_histories,
            ),
            functions=[function],
            function_call=function,
            timeout=timeout,
        )
        if "arguments" not in function_call_parsed:
            raise ValueError("No arguments returned")
        return str(function_call_parsed["arguments"]["query_to_search"])
    except Exception:
        return query


async def aget_json(
    query_template: PromptTemplate, **kwargs_to_format: str
) -> Optional[Any]:
    """Get json from query template and kwargs to format"""
    try:
        json_query = JSON_PATTERN.search(
            await Shared().llm.apredict(query_template.format(**kwargs_to_format))
        )
        if json_query is None:
            raise ValueError("Result is None")
        else:
            return orjson_loads(json_query.group())
    except Exception:
        return None
