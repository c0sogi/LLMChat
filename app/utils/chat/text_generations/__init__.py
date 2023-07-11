from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

from app.models.base_models import APIChatMessage, TextGenerationSettings
from app.models.completion_models import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
)

if TYPE_CHECKING:
    from app.models.llms import LLMModel


class BaseCompletionGenerator(ABC):
    """Base class for all completion generators."""

    user_role: str = "user"
    system_role: str = "system"

    user_input_role: str = "User"
    system_input_role: str = "System"

    ai_fallback_input_role: str = "Assistant"

    @abstractmethod
    def __del__(self):
        """Clean up resources."""
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, llm_model: "LLMModel"
    ) -> "BaseCompletionGenerator":
        """Load a pretrained model into RAM."""
        ...

    @abstractmethod
    def generate_completion(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Completion:
        """Generate a completion for a given prompt."""
        ...

    @abstractmethod
    def generate_completion_with_streaming(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Iterator[CompletionChunk]:
        """Generate a completion for a given prompt, yielding chunks of text as they are generated."""
        ...

    @abstractmethod
    def generate_chat_completion(
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> ChatCompletion:
        """Generate a completion for a given prompt."""
        ...

    @abstractmethod
    def generate_chat_completion_with_streaming(
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> Iterator[ChatCompletionChunk]:
        """Generate a completion for a given prompt, yielding chunks of text as they are generated."""
        ...

    @staticmethod
    def get_stop_strings(*roles: str) -> list[str]:
        """A helper method to generate stop strings for a given set of roles.
        Stop strings are required to stop text completion API from generating
        text that does not belong to the current chat turn.
        e.g. The common stop string is "### USER:", which can prevent ai from generating
        user's message itself."""

        prompt_stop = set()
        for role in roles:
            avoids = (
                f"{role}:",
                f"### {role}:",
                f"###{role}:",
            )
            prompt_stop.update(
                avoids,
                map(str.capitalize, avoids),
                map(str.upper, avoids),
                map(str.lower, avoids),
            )
        return list(prompt_stop)

    @classmethod
    def convert_messages_into_prompt(
        cls, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> str:
        """A helper method to convert list of messages into one text prompt."""
        ai_input_role: str = cls.ai_fallback_input_role
        chat_history: str = ""
        for message in messages:
            if message.role.lower() == cls.user_role:
                input_role = cls.user_input_role
            elif message.role.lower() == cls.system_role:
                input_role = cls.system_input_role
            else:
                input_role = ai_input_role = message.role
            chat_history += f"### {input_role}:{message.content}"

        prompt_stop: list[str] = cls.get_stop_strings(
            cls.user_input_role, cls.system_input_role, ai_input_role
        )
        if isinstance(settings.stop, str):
            settings.stop = prompt_stop + [settings.stop]
        elif isinstance(settings.stop, list):
            settings.stop = prompt_stop + settings.stop
        else:
            settings.stop = prompt_stop
        return chat_history + f"### {ai_input_role}:"

    @staticmethod
    def is_possible_to_generate_stops(
        decoded_text: str, stops: list[str]
    ) -> bool:
        """A helper method to check if the decoded text contains any of the stop tokens."""
        for stop in stops:
            if stop in decoded_text or any(
                [
                    decoded_text.endswith(stop[: i + 1])
                    for i in range(len(stop))
                ]
            ):
                return True
        return False

    @property
    @abstractmethod
    def llm_model(self) -> "LLMModel":
        """The LLM model used by this generator."""
        ...
