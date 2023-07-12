from typing import Optional

from app.models.chat_models import ResponseType
from app.models.function_calling.functions import FunctionCalls
from app.models.llms import OpenAIModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.messages.handler import MessageHandler
from app.utils.function_calling.query import aget_query_to_search


class BrowsingCommands:
    @staticmethod
    async def browse(
        user_query: str, /, buffer: BufferedUserContext, **kwargs
    ) -> tuple[Optional[str], ResponseType]:
        """Query LLM with duckduckgo browse results\n
        /browse <query>"""
        if user_query.startswith("/"):
            # User is trying to invoke another command.
            # Give control back to the command handler,
            # and let it handle the command.
            # e.g. `/browse /help` will invoke `/help` command
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
                    FunctionCalls.web_search
                ),
            )
            await MessageHandler.function_call(
                callback_name=FunctionCalls.web_search.__name__,
                callback_kwargs={"query_to_search": query_to_search},
                buffer=buffer,
            )
        else:
            # OpenAI models can invoke function call,
            # so let the AI decide whether to invoke function call
            function = FunctionCalls.get_function_call(
                FunctionCalls.web_search
            )
            buffer.optional_info["functions"] = [function]
            buffer.optional_info["function_call"] = function
            await MessageHandler.ai(buffer=buffer)

        # End of command
        return None, ResponseType.DO_NOTHING
