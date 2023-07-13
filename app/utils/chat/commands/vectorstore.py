from typing import Optional

from app.common.config import config
from app.common.lotties import Lotties
from app.models.chat_models import ResponseType, command_response
from app.models.function_calling.functions import FunctionCalls
from app.models.llms import OpenAIModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.vectorstore import VectorStoreManager
from app.utils.chat.messages.handler import MessageHandler
from app.utils.function_calling.query import aget_query_to_search
from app.viewmodels.status import UserStatus


class VectorstoreCommands:
    @staticmethod
    async def query(
        user_query: str, /, buffer: BufferedUserContext, **kwargs
    ) -> tuple[Optional[str], ResponseType]:
        """Query from redis vectorstore\n
        /query <query>"""
        if user_query.startswith("/"):
            # User is trying to invoke another command.
            # Give control back to the command handler,
            # and let it handle the command.
            # e.g. `/query /help` will invoke `/help` command
            return user_query, ResponseType.REPEAT_COMMAND

        # Save user query to buffer and database
        await MessageHandler.user(msg=user_query, buffer=buffer)

        if not isinstance(buffer.current_llm_model.value, OpenAIModel):
            # Non-OpenAI models can't invoke function call,
            # so we force function calling here
            query_to_search: str = await aget_query_to_search(
                buffer=buffer,
                query=user_query,
                function=FunctionCalls.get_function_call(
                    FunctionCalls.vectorstore_search
                ),
            )
            await MessageHandler.function_call(
                callback_name=FunctionCalls.vectorstore_search.__name__,
                callback_kwargs={"query_to_search": query_to_search},
                buffer=buffer,
            )
        else:
            # OpenAI models can invoke function call,
            # so let the AI decide whether to invoke function call
            function = FunctionCalls.get_function_call(
                FunctionCalls.vectorstore_search
            )
            buffer.optional_info["functions"] = [function]
            buffer.optional_info["function_call"] = function
            await MessageHandler.ai(buffer=buffer)

        # End of command
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
