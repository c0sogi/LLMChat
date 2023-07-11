from typing import Optional, Tuple

from app.common.config import config
from app.common.constants import QueryTemplates
from app.common.lotties import Lotties
from app.models.chat_models import ChatRoles, ResponseType, command_response
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.message import MessageManager
from app.utils.chat.managers.vectorstore import VectorStoreManager
from app.utils.chat.messages.handler import MessageHandler
from app.utils.chat.tokens import make_formatted_query
from app.utils.function_calling.callbacks.translate import translate_callback
from app.utils.function_calling.callbacks.vectorstore_search import (
    vectorstore_search_callback,
)
from app.viewmodels.status import UserStatus


class VectorstoreCommands:
    @staticmethod
    async def query(
        user_query: str, /, buffer: BufferedUserContext, **kwargs
    ) -> Tuple[Optional[str], ResponseType]:
        """Query from redis vectorstore\n
        /query <query>"""
        if user_query.startswith("/"):
            return user_query, ResponseType.REPEAT_COMMAND

        translate: Optional[str] = kwargs.get("translate", None)
        if translate:
            translate_chain_result: Optional[str] = await translate_callback(
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
        vectorstore_query_result: Optional[
            str
        ] = await vectorstore_search_callback(
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
            buffer=buffer,
            use_tight_token_limit=False,
        )
        try:
            await MessageHandler.ai(buffer=buffer)
        finally:
            if vectorstore_query_result is not None:
                await MessageManager.set_message_history_safely(
                    user_chat_context=buffer.current_user_chat_context,
                    role=ChatRoles.USER,
                    index=-1,
                    new_content=user_query,
                )
        return None, ResponseType.DO_NOTHING

    @staticmethod
    @command_response.send_message_and_stop
    async def embed(text_to_embed: str, /, buffer: BufferedUserContext) -> str:
        """Embed the text and save its vectors in the redis vectorstore.\n
        /embed <text_to_embed>"""
        await VectorStoreManager.create_documents(
            text=text_to_embed, collection_name=buffer.user_id
        )
        return Lotties.OK.format("Embedding successful!")

    @staticmethod
    @command_response.send_message_and_stop
    async def share(text_to_embed: str, /) -> str:
        """Embed the text and save its vectors in the redis vectorstore. This index is shared for everyone.\n
        /share <text_to_embed>"""
        await VectorStoreManager.create_documents(
            text=text_to_embed, collection_name=config.shared_vectorestore_name
        )
        return Lotties.OK.format(
            "Embedding successful!\nThis data will be shared for everyone."
        )

    @staticmethod
    @command_response.send_message_and_stop
    async def drop(buffer: BufferedUserContext) -> str:
        """Drop the index from the redis vectorstore.\n
        /drop"""
        dropped_index: list[str] = []
        if await VectorStoreManager.delete_collection(
            collection_name=buffer.user_id
        ):
            dropped_index.append(buffer.user_id)
        if (
            buffer.user.status is UserStatus.admin
            and await VectorStoreManager.delete_collection(
                collection_name=config.shared_vectorestore_name,
            )
        ):
            dropped_index.append(config.shared_vectorestore_name)
        if not dropped_index:
            return "No index dropped."
        return f"Index dropped: {', '.join(dropped_index)}"
