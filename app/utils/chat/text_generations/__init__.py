from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Iterator, Self

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
    @abstractproperty
    def llm_model(self) -> "LLMModel":
        """The LLM model used by this generator."""
        ...

    @abstractmethod
    def __del__(self) -> None:
        """Clean up resources."""
        ...

    @abstractclassmethod
    def from_pretrained(cls, llm_model: "LLMModel") -> Self:
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

    def convert_messages_into_prompt(
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> str:
        chat_history = "".join(
            f'### {"Human" if message.role == "user" else "Assistant"}:{message.content}'
            for message in messages
        )
        prompt_stop = ["### Assistant:", "### Human:"]
        if isinstance(settings.stop, str):
            settings.stop = prompt_stop + [settings.stop]
        elif isinstance(settings.stop, list):
            settings.stop = prompt_stop + settings.stop
        else:
            settings.stop = prompt_stop
        return chat_history + "### Assistant:"
