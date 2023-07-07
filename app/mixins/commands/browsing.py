from typing import Optional

from app.models.chat_models import ResponseType
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.commands.browse import browse


class BrowsingCommandMixin:
    @staticmethod
    async def browse(
        user_query: str, /, buffer: BufferedUserContext, **kwargs
    ) -> tuple[Optional[str], ResponseType]:
        """Query LLM with duckduckgo browse results\n
        /browse <query>"""
        return await browse(user_query, buffer=buffer, **kwargs)

    # @staticmethod
    # @command_response.send_message_and_stop
    # async def browsesearx(query: str, /) -> str:
    #     """Search web for the query, with searxNG\n
    #     /browsesearx <query>"""
    #     return await Shared().searx.arun(query)