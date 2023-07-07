from typing import Optional
from app.models.chat_models import command_response
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.commands.summarize import summarize


class SummarizeCommandsMixin:
    @staticmethod
    @command_response.send_message_and_stop
    async def summarize(
        to_summarize: Optional[str], /, buffer: BufferedUserContext
    ) -> str:
        """Summarize the conversation\n
        /summarize"""
        return await summarize(to_summarize, buffer=buffer)
