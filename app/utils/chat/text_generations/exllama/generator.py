"""Wrapper for exllama to generate text completions."""
import sys
from pathlib import Path

from torch import cuda

from app.models.llm_tokenizers import ExllamaTokenizer
from app.utils.chat.text_generations.path import resolve_model_path_to_posix

assert cuda.is_available()
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from app.utils.logger import ApiLogger

from .. import BaseCompletionGenerator

sys.path.insert(0, str(Path("repositories/exllama")))
from repositories.exllama.generator import ExLlamaGenerator
from repositories.exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from repositories.exllama.tokenizer import ExLlamaTokenizer

if TYPE_CHECKING:
    from app.models.base_models import APIChatMessage, TextGenerationSettings
    from app.models.completion_models import (
        ChatCompletion,
        ChatCompletionChunk,
        Completion,
        CompletionChunk,
    )
    from app.models.llms import ExllamaModel

logger = ApiLogger("||🦙 exllama.generator||")


def _make_config(llm_model: "ExllamaModel") -> ExLlamaConfig:
    """Create a config object for the ExLlama model."""
    model_folder_path = Path(
        resolve_model_path_to_posix(
            llm_model.model_path,
            default_relative_directory="llama_models/gptq",
        ),
    )
    config = ExLlamaConfig((model_folder_path / "config.json").as_posix())

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
    config.model_path = model_file_found[-1].as_posix()  # type: ignore
    config.max_seq_len = (
        llm_model.max_total_tokens
    )  # Can also be increased, ideally while also using compress_pos_emn and a compatible model/LoRA
    config.max_input_len = (
        llm_model.max_total_tokens
    )  # Maximum length of input IDs in a single forward pass,
    # sequences longer than this will be processed in multiple steps.
    config.max_attention_size = (
        2048**2
    )  # Sequences will be processed in chunks to keep the size of the attention weights matrix <= this
    config.compress_pos_emb = llm_model.compress_pos_emb
    config.gpu_peer_fix = llm_model.gpu_peer_fix
    config.auto_map = llm_model.auto_map
    config.matmul_fused_remap = llm_model.matmul_fused_remap
    config.fused_mlp_thd = llm_model.fused_mlp_thd
    config.sdp_thd = llm_model.sdp_thd
    config.fused_attn = llm_model.fused_attn
    config.matmul_fused_remap = llm_model.matmul_fused_remap
    config.rmsnorm_no_half2 = llm_model.rmsnorm_no_half2
    config.rope_no_half2 = llm_model.rope_no_half2
    config.matmul_fused_remap = llm_model.matmul_fused_remap
    config.silu_no_half2 = llm_model.silu_no_half2
    config.concurrent_streams = llm_model.concurrent_streams
    return config


def _make_generator(
    model: ExLlama,
    tokenizer: ExLlamaTokenizer,
    cache: ExLlamaCache,
    settings: "TextGenerationSettings",
) -> ExLlamaGenerator:
    """Make a generator object for the ExLlama model."""
    generator = ExLlamaGenerator(model=model, tokenizer=tokenizer, cache=cache)
    generator.settings.temperature = settings.temperature
    generator.settings.top_p = settings.top_p
    generator.settings.top_k = settings.top_k
    generator.settings.typical = settings.typical_p
    generator.settings.token_repetition_penalty_max = settings.repeat_penalty
    if settings.ban_eos_token and tokenizer.eos_token_id is not None:
        generator.disallow_tokens([tokenizer.eos_token_id])
    return generator


class ExllamaCompletionGenerator(BaseCompletionGenerator):
    config: Optional[ExLlamaConfig] = None
    model: Optional[ExLlama] = None
    cache: Optional[ExLlamaCache] = None
    tokenizer: Optional[ExLlamaTokenizer] = None
    generator: Optional[ExLlamaGenerator] = None
    _llm_model: Optional["ExllamaModel"] = None
    _completion_status: dict[
        str, int
    ] = {}  # key: completion_id, value: number of completion tokens

    def __del__(self) -> None:
        if self.model is not None:
            self.model.free_unmanaged()
        del self.model
        del self.cache
        del self.tokenizer
        del self.generator
        self.model = None
        self.cache = None
        self.tokenizer = None
        self.generator = None
        cuda.empty_cache()
        print("🗑️ ExllamaCompletionGenerator deleted")

    @property
    def llm_model(self) -> "ExllamaModel":
        assert self._llm_model is not None
        return self._llm_model

    @classmethod
    def from_pretrained(cls, llm_model: "ExllamaModel") -> "ExllamaCompletionGenerator":
        result = cls()
        result.config = _make_config(llm_model)
        result.tokenizer = llm_model.tokenizer.tokenizer
        result.model = ExLlama(result.config)
        result.cache = ExLlamaCache(result.model)
        result._llm_model = llm_model
        return result

    def _generate_text(self, prompt: str, settings: "TextGenerationSettings") -> str:
        return "".join(self._generate_text_with_streaming(prompt, settings=settings))

    def _generate_text_with_streaming(
        self, prompt: str, settings: "TextGenerationSettings"
    ) -> Iterator[str]:
        assert (
            self.model is not None
            and self.tokenizer is not None
            and self.cache is not None
        )

        # Make sure that the stop token is a list
        if isinstance(settings.stop, str):
            stops = [settings.stop]
        elif isinstance(settings.stop, list):
            stops = settings.stop
        else:
            stops = []

        self.generator = _make_generator(
            model=self.model,
            tokenizer=self.tokenizer,
            cache=self.cache,
            settings=settings,
        )

        # Start generation
        self.generator.end_beam_search()
        ids = self.generator.tokenizer.encode(prompt)
        self.generator.gen_begin_reuse(ids)
        initial_len = self.generator.sequence[0].shape[0]
        has_leading_space: bool = False
        text_cursor: int = 0
        n_completion_tokens: int = 0
        for n_completion_tokens in range(1, settings.max_tokens + 1):
            token = self.generator.gen_single_token()
            if token.item() == self.generator.tokenizer.eos_token_id:
                return
            if (
                n_completion_tokens == 0
                and self.generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith(
                    "▁"
                )
            ):
                has_leading_space = True

            decoded_text = str(
                self.generator.tokenizer.decode(
                    self.generator.sequence[0][initial_len:]
                )
            )
            if has_leading_space:
                decoded_text = " " + decoded_text
            if self.is_possible_to_generate_stops(decoded_text, stops=stops):
                for stop in stops:
                    if stop in decoded_text:
                        return
                continue
            text_piece = decoded_text[text_cursor:]
            if "�" in text_piece:
                continue
            yield text_piece
            text_cursor += len(text_piece)
        self._completion_status[settings.completion_id] = n_completion_tokens

    def generate_completion_with_streaming(
        self, prompt: str, settings: "TextGenerationSettings"
    ) -> Iterator["CompletionChunk"]:
        assert self.config is not None and self.tokenizer is not None
        completion_id: str = settings.completion_id
        model_path: str = str(self.config.model_path)
        last_token: Optional[str] = None
        generated_text: str = ""
        for token in self._generate_text_with_streaming(prompt, settings=settings):
            generated_text += token
            if last_token is not None:
                yield self.make_completion_chunk(
                    completion_id=completion_id,
                    model=model_path,
                    generated_text=last_token,
                    finish_reason="null",
                )
            last_token = token
        yield self.make_completion_chunk(
            completion_id=completion_id,
            model=model_path,
            generated_text=last_token if last_token is not None else "",
            finish_reason="length"
            if self._completion_status.get(
                completion_id, self.tokenizer.encode(generated_text).shape[1]
            )
            >= settings.max_tokens
            else "stop",
        )

    def generate_completion(
        self, prompt: str, settings: "TextGenerationSettings"
    ) -> "Completion":
        assert self.tokenizer is not None and self.config is not None
        completion_id: str = settings.completion_id
        generated_text: str = self._generate_text(prompt, settings=settings)
        n_prompt_tokens: int = self.tokenizer.encode(prompt).shape[1]
        n_completion_tokens: int = self._completion_status.get(
            completion_id, self.tokenizer.encode(generated_text).shape[1]
        )
        return self.make_completion(
            completion_id=completion_id,
            model=str(self.config.model_path),
            generated_text=generated_text,
            n_prompt_tokens=n_prompt_tokens,
            n_completion_tokens=n_completion_tokens,
            finish_reason="length"
            if n_completion_tokens >= settings.max_tokens
            else "stop",
        )

    def generate_chat_completion_with_streaming(
        self, messages: list["APIChatMessage"], settings: "TextGenerationSettings"
    ) -> Iterator["ChatCompletionChunk"]:
        assert self.config is not None and self.tokenizer is not None
        completion_id: str = settings.completion_id
        prompt = self.convert_messages_into_prompt(messages, settings=settings)
        model_path: str = str(self.config.model_path)
        last_token: Optional[str] = None
        generated_text: str = ""
        for token in self._generate_text_with_streaming(prompt, settings=settings):
            generated_text += token
            if last_token is not None:
                yield self.make_chat_completion_chunk(
                    completion_id=completion_id,
                    model=model_path,
                    generated_text=last_token,
                    finish_reason="null",
                )
            last_token = token
        yield self.make_chat_completion_chunk(
            completion_id=completion_id,
            model=model_path,
            generated_text=last_token if last_token is not None else "",
            finish_reason="length"
            if self._completion_status.get(
                completion_id, self.tokenizer.encode(generated_text).shape[1]
            )
            else "stop",
        )

    def generate_chat_completion(
        self, messages: list["APIChatMessage"], settings: "TextGenerationSettings"
    ) -> "ChatCompletion":
        assert self.tokenizer is not None and self.config is not None
        completion_id: str = settings.completion_id
        prompt = self.convert_messages_into_prompt(messages, settings=settings)
        generated_text: str = self._generate_text(prompt, settings=settings)
        n_prompt_tokens: int = self.tokenizer.encode(prompt).shape[1]
        n_completion_tokens: int = self._completion_status.get(
            completion_id, self.tokenizer.encode(generated_text).shape[1]
        )
        return self.make_chat_completion(
            completion_id=completion_id,
            model=str(self.config.model_path),
            generated_text=generated_text,
            n_prompt_tokens=n_prompt_tokens,
            n_completion_tokens=n_completion_tokens,
            finish_reason="length"
            if n_completion_tokens >= settings.max_tokens
            else "stop",
        )


if __name__ == "__main__":
    from app.models.base_models import TextGenerationSettings
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
