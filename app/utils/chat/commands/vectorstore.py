from typing import Optional, Tuple

from app.common.constants import (
    QueryTemplates,
)
from app.models.chat_models import ChatRoles, ResponseType
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.chains.translate import translate_chain
from app.utils.chat.chains.vectorstore_query import vectorstore_query_chain
from app.utils.chat.messages.handler import MessageHandler
from app.utils.chat.managers.message import MessageManager
from app.utils.chat.tokens import make_formatted_query


async def query(
    user_query: str, /, buffer: BufferedUserContext, **kwargs
) -> Tuple[Optional[str], ResponseType]:
    """Query from redis vectorstore\n
    /query <query>"""
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
    vectorstore_query_result: Optional[str] = await vectorstore_query_chain(
        buffer=buffer,
        query=user_query,
        finish=True,
        wait_next_query=True,
        show_result=False,
    )
    if vectorstore_query_result:
        query_to_send: str = make_formatted_query(
            user_chat_context=buffer.current_user_chat_context,
            question=user_query,
            context=vectorstore_query_result,
            query_template=QueryTemplates.CONTEXT_QUESTION__CONTEXT_ONLY,
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
        if vectorstore_query_result is not None:
            await MessageManager.set_message_history_safely(
                user_chat_context=buffer.current_user_chat_context,
                role=ChatRoles.USER,
                index=-1,
                new_content=user_query,
            )
    return None, ResponseType.DO_NOTHING
