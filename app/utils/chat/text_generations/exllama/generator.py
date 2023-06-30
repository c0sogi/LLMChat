from gc import collect
import sys
from pathlib import Path
from time import time
from uuid import uuid4

from torch import cuda
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
from app.models.llm_tokenizers import ExllamaTokenizer

from app.models.llms import ExllamaModel
from app.utils.chat.text_generations.path import resolve_model_path_to_posix

assert cuda.is_available()
from pathlib import Path
from typing import Iterator, Optional, Self

from app.utils.logger import ApiLogger

from .. import BaseCompletionGenerator

sys.path.insert(0, str(Path("repositories/exllama")))
from repositories.exllama.generator import ExLlamaGenerator
from repositories.exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from repositories.exllama.tokenizer import ExLlamaTokenizer

logger = ApiLogger("||🦙 exllama.generator||")


def _is_possible_to_generate_stops(decoded_text: str, stops: list[str]) -> bool:
    """Checks if the decoded text contains any of the stop tokens."""
    for avoid in stops:
        avoid = avoid.upper()
        if avoid in decoded_text or any(
            [decoded_text.endswith(avoid[: i + 1]) for i in range(len(avoid))]
        ):
            return True
    return False


class ExllamaCompletionGenerator(BaseCompletionGenerator):
    config: Optional[ExLlamaConfig] = None
    model: Optional[ExLlama] = None
    cache: Optional[ExLlamaCache] = None
    tokenizer: Optional[ExLlamaTokenizer] = None
    generator: Optional[ExLlamaGenerator] = None
    _llm_model: Optional[ExllamaModel] = None

    def __del__(self) -> None:
        if self.model is not None:
            self.model.free_unmanaged()
        del self.model
        del self.cache
        del self.tokenizer
        del self.generator
        collect()

    def _apply_settings_to_generator(self, settings: TextGenerationSettings) -> None:
        assert self.generator is not None and self.tokenizer is not None
        self.generator.settings.temperature = settings.temperature
        self.generator.settings.top_p = settings.top_p
        self.generator.settings.top_k = settings.top_k
        self.generator.settings.typical = settings.typical_p
        self.generator.settings.token_repetition_penalty_max = settings.repeat_penalty
        if settings.ban_eos_token:
            self.generator.disallow_tokens([self.tokenizer.eos_token_id])

    @property
    def llm_model(self) -> ExllamaModel:
        assert self._llm_model is not None
        return self._llm_model

    @classmethod
    def from_pretrained(cls, llm_model: ExllamaModel) -> Self:
        model_folder_path = Path(
            resolve_model_path_to_posix(
                llm_model.model_path,
                default_relative_directory="llama_models/gptq",
            ),
        )

        # Find the model checkpoint
        model_file_found: list[Path] = []
        for ext in (".safetensors", ".pt", ".bin"):
            model_file_found.extend(model_folder_path.glob(f"*{ext}"))
            if model_file_found:
                if len(model_file_found) > 1:
                    logger.warning(
                        f"More than one {ext} model has been found. The last one will be selected. It could be wrong."
                    )

                break
        if not model_file_found:
            raise FileNotFoundError(f"No model has been found in {model_folder_path}.")
        config = ExLlamaConfig((model_folder_path / "config.json").as_posix())
        config.model_path = model_file_found[-1].as_posix()  # type: ignore
        model = ExLlama(config)

        result = cls()
        result.config = config
        result.model = model
        result.tokenizer = llm_model.tokenizer.tokenizer
        result.cache = ExLlamaCache(model)
        result._llm_model = llm_model
        return result

    def _generate_text(self, prompt: str, settings: TextGenerationSettings) -> str:
        return "".join(self._generate_text_with_streaming(prompt, settings=settings))

    def _generate_text_with_streaming(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Iterator[str]:
        assert self.tokenizer is not None
        if isinstance(settings.stop, str):
            stops = [settings.stop]
        elif isinstance(settings.stop, list):
            stops = settings.stop
        else:
            stops = []
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        self._apply_settings_to_generator(settings)

        self.generator.end_beam_search()
        ids = self.generator.tokenizer.encode(prompt)
        self.generator.gen_begin_reuse(ids)
        initial_len = self.generator.sequence[0].shape[0]
        has_leading_space: bool = False
        text_cursor: int = 0
        for i in range(settings.max_tokens):
            token = self.generator.gen_single_token()
            if token.item() == self.generator.tokenizer.eos_token_id:
                return
            if i == 0 and self.generator.tokenizer.tokenizer.IdToPiece(
                int(token)
            ).startswith("▁"):
                has_leading_space = True

            decoded_text = str(
                self.generator.tokenizer.decode(
                    self.generator.sequence[0][initial_len:]
                )
            )
            if has_leading_space:
                decoded_text = " " + decoded_text
            if _is_possible_to_generate_stops(decoded_text, stops=stops):
                for stop in stops:
                    if stop in decoded_text:
                        return
                continue
            text_piece = decoded_text[text_cursor:]
            if "�" in text_piece:
                continue
            yield text_piece
            text_cursor += len(text_piece)

    def generate_completion_with_streaming(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Iterator[CompletionChunk]:
        assert self.config is not None
        completion_id: str = f"cmpl-{str(uuid4())}"
        last_token: Optional[str] = None
        n_tokens: int = 0
        for token in self._generate_text_with_streaming(prompt, settings=settings):
            n_tokens += 1
            if last_token is not None:
                yield CompletionChunk(
                    id=completion_id,
                    object="text_completion",
                    created=int(time()),
                    model=str(self.config.model_path),
                    choices=[
                        CompletionChoice(
                            text=last_token,
                            index=0,
                            logprobs=None,
                            finish_reason="null",
                        )
                    ],
                )
            last_token = token
        yield CompletionChunk(
            id=completion_id,
            object="text_completion",
            created=int(time()),
            model=str(self.config.model_path),
            choices=[
                CompletionChoice(
                    text=last_token if last_token is not None else "",
                    index=0,
                    logprobs=None,
                    finish_reason="length"
                    if n_tokens >= settings.max_tokens
                    else "stop",
                )
            ],
        )

    def generate_completion(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Completion:
        assert self.tokenizer is not None and self.config is not None
        completion_id: str = f"cmpl-{str(uuid4())}"
        generated_text: str = self._generate_text(prompt, settings=settings)
        prompt_tokens: int = self.tokenizer.encode(prompt).shape[1]
        completion_tokens: int = self.tokenizer.encode(generated_text).shape[1]

        return Completion(
            id=completion_id,
            object="text_completion",
            created=int(time()),
            model=str(self.config.model_path),
            choices=[
                CompletionChoice(
                    text=generated_text,
                    index=0,
                    logprobs=None,
                    finish_reason="length"
                    if completion_tokens >= settings.max_tokens
                    else "stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def generate_chat_completion_with_streaming(
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> Iterator[ChatCompletionChunk]:
        assert self.config is not None
        prompt = self.convert_messages_into_prompt(messages, settings=settings)
        completion_id: str = f"cmpl-{str(uuid4())}"
        last_token: Optional[str] = None
        n_tokens: int = 0
        for token in self._generate_text_with_streaming(prompt, settings=settings):
            n_tokens += 1
            if last_token is not None:
                yield ChatCompletionChunk(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=int(time()),
                    model=str(self.config.model_path),
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(
                                content=last_token,
                            ),
                            finish_reason="null",
                        )
                    ],
                )
            last_token = token
        yield ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=int(time()),
            model=str(self.config.model_path),
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(
                        content=last_token if last_token is not None else "",
                    ),
                    finish_reason="length"
                    if n_tokens >= settings.max_tokens
                    else "stop",
                )
            ],
        )

    def generate_chat_completion(
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> ChatCompletion:
        assert self.tokenizer is not None and self.config is not None
        prompt = self.convert_messages_into_prompt(messages, settings=settings)
        completion_id: str = f"cmpl-{str(uuid4())}"
        generated_text: str = self._generate_text(prompt, settings=settings)
        prompt_tokens: int = self.tokenizer.encode(prompt).shape[1]
        completion_tokens: int = self.tokenizer.encode(generated_text).shape[1]

        return ChatCompletion(
            id=completion_id,
            object="chat.completion",
            created=int(time()),
            model=str(self.config.model_path),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=generated_text,
                    ),
                    finish_reason="length"
                    if completion_tokens >= settings.max_tokens
                    else "stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


if __name__ == "__main__":
    from app.models.llms import ExllamaModel

    exllama_model = ExllamaModel(
        model_path="orca_mini_7b",
        name="orca_mini_7b",
        max_total_tokens=1024,
        max_tokens_per_request=1024,
        token_margin=8,
        tokenizer=ExllamaTokenizer("orca_mini_7b"),
    )
    generator: ExllamaCompletionGenerator = ExllamaCompletionGenerator.from_pretrained(
        exllama_model
    )
    print("\n\n")
    for token in generator.generate_completion_with_streaming(
        "Hello. My name is", settings=TextGenerationSettings()
    ):
        print(token, flush=True)

    print("\n\n")
    print(
        generator.generate_completion(
            "Hello. My name is", settings=TextGenerationSettings()
        )
    )
    del generator
