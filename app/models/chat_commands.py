from typing import Callable

from app.utils.chat.commands.browsing import BrowsingCommands
from app.utils.chat.commands.core import CoreCommands
from app.utils.chat.commands.llm_parameter import LLMParameterCommands
from app.utils.chat.commands.prompt import PromptCommands
from app.utils.chat.commands.server import ServerCommands
from app.utils.chat.commands.summarize import SummarizeCommands
from app.utils.chat.commands.testing import TestingCommands
from app.utils.chat.commands.vectorstore import VectorstoreCommands

from .chat_models import command_response


class ChatCommands(
    CoreCommands,
    VectorstoreCommands,
    PromptCommands,
    BrowsingCommands,
    LLMParameterCommands,
    ServerCommands,
    TestingCommands,
    SummarizeCommands,
):
    @classmethod
    def _find_callback_with_command(cls, command: str) -> Callable:
        return getattr(cls, command, cls.not_existing_callback)

    @staticmethod
    @command_response.send_message_and_stop
    def not_existing_callback() -> str:  # callback for not existing command
        return "Sorry, I don't know what you mean by..."
