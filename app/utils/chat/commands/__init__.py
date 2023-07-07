from typing import Callable

from app.mixins.commands.browsing import BrowsingCommandMixin
from app.mixins.commands.core import CoreCommandsMixin
from app.mixins.commands.llm_parameter import LLMParameterCommandsMixin
from app.mixins.commands.prompt import PromptCommandsMixin
from app.mixins.commands.server import ServerCommandsMixin
from app.mixins.commands.summarize import SummarizeCommandsMixin
from app.mixins.commands.testing import TestingCommandsMixin
from app.mixins.commands.vectorstore import VectorstoreCommandsMixin
from app.models.chat_models import (
    command_response,
)


class ChatCommands(
    CoreCommandsMixin,
    VectorstoreCommandsMixin,
    PromptCommandsMixin,
    BrowsingCommandMixin,
    LLMParameterCommandsMixin,
    ServerCommandsMixin,
    TestingCommandsMixin,
    SummarizeCommandsMixin,
):
    @classmethod
    def _find_callback_with_command(cls, command: str) -> Callable:
        return getattr(cls, command, cls.not_existing_callback)

    @staticmethod
    @command_response.send_message_and_stop
    def not_existing_callback() -> str:  # callback for not existing command
        return "Sorry, I don't know what you mean by..."
