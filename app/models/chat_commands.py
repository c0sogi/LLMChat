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


class ChatCommandsMetaClass(type):
    """Metaclass for ChatCommands class.
    It is used to automatically create list of special commands.
    """

    special_commands: list[str]

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls.special_commands = [
            callback_name
            for callback_name in dir(CoreCommands)
            if not callback_name.startswith("_")
        ]


class ChatCommands(
    CoreCommands,
    VectorstoreCommands,
    PromptCommands,
    BrowsingCommands,
    LLMParameterCommands,
    ServerCommands,
    TestingCommands,
    SummarizeCommands,
    metaclass=ChatCommandsMetaClass,
):
    @classmethod
    def find_callback_with_command(cls, command: str) -> Callable:
        found_callback: Callable = getattr(cls, command)
        if found_callback is None or not callable(found_callback):
            found_callback = cls.not_existing_callback
        return found_callback

    @staticmethod
    @command_response.send_message_and_stop
    def not_existing_callback() -> str:  # callback for not existing command
        return "Sorry, I don't know what you mean by..."
