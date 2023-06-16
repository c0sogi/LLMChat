from typing import Optional, Tuple

from app.common.constants import (
    QueryTemplates,
)
from app.models.chat_models import ChatRoles, ResponseType
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.chains.full_browsing import full_web_browsing_chain
from app.utils.chat.chains.translate import translate_chain
from app.utils.chat.messages.handler import MessageHandler
from app.utils.chat.managers.message import MessageManager
from app.utils.chat.tokens import get_token_limit_with_n_messages, make_formatted_query
from app.utils.logger import ApiLogger


async def browse(
    user_query: str, /, buffer: BufferedUserContext, **kwargs
) -> Tuple[Optional[str], ResponseType]:
    """Query LLM with duckduckgo browse results\n
    /browse <query>"""
    if user_query.startswith("/"):
        return user_query, ResponseType.REPEAT_COMMAND

    translate: Optional[str] = kwargs.get("translate", None)
    if translate:
        translate_chain_result: Optional[str] = await translate_chain(
            buffer=buffer,
            query=user_query,
            finish=False,
            wait_next_query=False,
            show_result=True,
            src_lang=translate,
            trg_lang="en",
        )
        if translate_chain_result is not None:
            user_query = translate_chain_result
    browsing_result: Optional[str] = await full_web_browsing_chain(
        buffer=buffer,
        query=user_query,
        finish=True,
        wait_next_query=True,
        show_result=False,
    )
    if browsing_result:
        query_to_send: str = make_formatted_query(
            user_chat_context=buffer.current_user_chat_context,
            question=user_query,
            context=browsing_result,
            query_template=QueryTemplates.CONTEXT_QUESTION__DEFAULT,
        )
    else:
        query_to_send: str = user_query

    await MessageHandler.user(
        msg=query_to_send,
        translate=None,
        buffer=buffer,
        use_tight_token_limit=False,
    )
    try:
        await MessageHandler.ai(
            translate=translate,
            buffer=buffer,
        )
    finally:
        if browsing_result is not None:
            await MessageManager.set_message_history_safely(
                user_chat_context=buffer.current_user_chat_context,
                role=ChatRoles.USER,
                index=-1,
                new_content=user_query,
            )
    return None, ResponseType.DO_NOTHING
