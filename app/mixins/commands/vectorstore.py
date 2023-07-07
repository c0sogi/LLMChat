from typing import Optional

from app.common.config import config
from app.common.lotties import Lotties
from app.models.chat_models import (
    ResponseType,
    command_response,
)
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.commands.vectorstore import query
from app.utils.chat.managers.vectorstore import VectorStoreManager
from app.viewmodels.status import UserStatus


class VectorstoreCommandsMixin:
    @staticmethod
    async def query(
        user_query: str, /, buffer: BufferedUserContext, **kwargs
    ) -> tuple[Optional[str], ResponseType]:
        """Query from redis vectorstore\n
        /query <query>"""
        return await query(user_query, buffer=buffer, **kwargs)

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
