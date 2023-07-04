from abc import ABC, abstractmethod
from time import time
from typing import TYPE_CHECKING, Iterator

from app.models.base_models import APIChatMessage, TextGenerationSettings
from app.models.completion_models import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
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
    def __del__(self) -> None:
        """Clean up resources."""
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, llm_model: "LLMModel") -> "BaseCompletionGenerator":
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
    def is_possible_to_generate_stops(decoded_text: str, stops: list[str]) -> bool:
        """A helper method to check if the decoded text contains any of the stop tokens."""
        for stop in stops:
            if stop in decoded_text or any(
                [decoded_text.endswith(stop[: i + 1]) for i in range(len(stop))]
            ):
                return True
        return False

    @staticmethod
    def make_chat_completion(
        completion_id: str,
        model: str,
        generated_text: str,
        finish_reason: str,
        n_prompt_tokens: int,
        n_completion_tokens: int,
    ) -> ChatCompletion:
        """A helper method to make a chat completion."""
        return ChatCompletion(
            id=completion_id,
            object="chat.completion",
            created=int(time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=generated_text,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=n_prompt_tokens,
                completion_tokens=n_completion_tokens,
                total_tokens=n_prompt_tokens + n_completion_tokens,
            ),
        )

    @staticmethod
    def make_chat_completion_chunk(
        completion_id: str,
        model: str,
        generated_text: str,
        finish_reason: str,
    ) -> ChatCompletionChunk:
        """A helper method to make a chat completion chunk."""
        return ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=int(time()),
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(
                        content=generated_text,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

    @staticmethod
    def make_completion_chunk(
        completion_id: str,
        model: str,
        generated_text: str,
        finish_reason: str,
    ) -> CompletionChunk:
        """A helper method to make a completion chunk."""
        return CompletionChunk(
            id=completion_id,
            object="text_completion",
            created=int(time()),
            model=model,
            choices=[
                CompletionChoice(
                    text=generated_text,
                    index=0,
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
        )

    @staticmethod
    def make_completion(
        completion_id: str,
        model: str,
        generated_text: str,
        finish_reason: str,
        n_prompt_tokens: int,
        n_completion_tokens: int,
    ) -> Completion:
        """A helper method to make a completion."""
        return Completion(
            id=completion_id,
            object="text_completion",
            created=int(time()),
            model=model,
            choices=[
                CompletionChoice(
                    text=generated_text,
                    index=0,
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=n_prompt_tokens,
                completion_tokens=n_completion_tokens,
                total_tokens=n_prompt_tokens + n_completion_tokens,
            ),
        )

    @property
    @abstractmethod
    def llm_model(self) -> "LLMModel":
        """The LLM model used by this generator."""
        ...

    # def convert_messages_into_prompt(
    #     self, messages: list[APIChatMessage], settings: TextGenerationSettings
    # ) -> str:
    #     chat_history = "".join(
    #         f'### {"Human" if message.role == "user" else "Assistant"}:{message.content}'
    #         for message in messages
    #     )
    #     prompt_stop = ["### Assistant:", "### Human:"]
    #     if isinstance(settings.stop, str):
    #         settings.stop = prompt_stop + [settings.stop]
    #     elif isinstance(settings.stop, list):
    #         settings.stop = prompt_stop + settings.stop
    #     else:
    #         settings.stop = prompt_stop
    #     return chat_history + "### Assistant:"
