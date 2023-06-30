from gc import collect
import sys
from pathlib import Path
from app.models.base_models import APIChatMessage, TextGenerationSettings
from app.models.completion_models import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
)
from app.models.llms import LlamaCppModel

from pathlib import Path
from typing import Iterator, Optional, Self
from app.utils.chat.text_generations.path import resolve_model_path_to_posix

from app.utils.logger import ApiLogger

from .. import BaseCompletionGenerator

sys.path.insert(0, str(Path("repositories/llama_cpp")))
try:
    from repositories.llama_cpp.llama_cpp.server import app as llama_cpp_server
    from repositories.llama_cpp.llama_cpp.server.app import llama_cpp

    print("🦙 llama-cpp-python repository found!")
except Exception as e:
    ApiLogger.cwarning(
        "🦙 Could not import llama-cpp-python repository. "
        "Please pull the repository with: "
        "git submodule update --recursive\n"
        f"[[[ Received error: {e} ]]]\n"
        f"In {__file__} with name {__name__}\n",
        "yellow",
    )
    from llama_cpp.server import app as llama_cpp_server
    from llama_cpp.server.app import llama_cpp


logger = ApiLogger("||🦙 llama_cpp.generator||")


def _create_completion(
    client: llama_cpp.Llama, prompt: str, stream: bool, settings: TextGenerationSettings
) -> Completion | Iterator[CompletionChunk]:
    return client.create_completion(  # type: ignore
        stream=stream,
        prompt=prompt,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        logprobs=settings.logprobs,
        echo=settings.echo,
        stop=settings.stop,
        frequency_penalty=settings.frequency_penalty,
        presence_penalty=settings.presence_penalty,
        repeat_penalty=settings.repeat_penalty,
        top_k=settings.top_k,
        tfs_z=settings.tfs_z,
        mirostat_mode=settings.mirostat_mode,
        mirostat_tau=settings.mirostat_tau,
        mirostat_eta=settings.mirostat_eta,
        logits_processor=llama_cpp.LogitsProcessorList(
            [
                llama_cpp_server.make_logit_bias_processor(
                    client,
                    settings.logit_bias,
                    settings.logit_bias_type,
                ),
            ]
        )
        if settings.logit_bias is not None
        else None,
    )


def _create_chat_completion(
    client: llama_cpp.Llama,
    messages: list[APIChatMessage],
    stream: bool,
    settings: TextGenerationSettings,
) -> ChatCompletion | Iterator[llama_cpp.ChatCompletionChunk]:
    return client.create_chat_completion(
        stream=stream,
        messages=[
            llama_cpp.ChatCompletionMessage(**message.dict()) for message in messages
        ],
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        stop=settings.stop,
        frequency_penalty=settings.frequency_penalty,
        presence_penalty=settings.presence_penalty,
        repeat_penalty=settings.repeat_penalty,
        top_k=settings.top_k,
        tfs_z=settings.tfs_z,
        mirostat_mode=settings.mirostat_mode,
        mirostat_tau=settings.mirostat_tau,
        mirostat_eta=settings.mirostat_eta,
        logits_processor=llama_cpp.LogitsProcessorList(
            [
                llama_cpp_server.make_logit_bias_processor(
                    client,
                    settings.logit_bias,
                    settings.logit_bias_type,
                ),
            ]
        )
        if settings.logit_bias is not None
        else None,
    )


class LlamaCppCompletionGenerator(BaseCompletionGenerator):
    generator: Optional[Iterator[CompletionChunk | ChatCompletionChunk]] = None
    client: Optional[llama_cpp.Llama] = None
    _llm_model: Optional[LlamaCppModel] = None

    def __del__(self) -> None:
        if self.client is not None and self.client.ctx is not None:
            llama_cpp.llama_free(self.client.ctx)
            self.client.set_cache(None)
        del self.client
        del self.generator
        collect()

    @property
    def llm_model(self) -> LlamaCppModel:
        assert self._llm_model is not None
        return self._llm_model

    @classmethod
    def from_pretrained(cls, llm_model: LlamaCppModel) -> Self:
        client = llama_cpp.Llama(
            model_path=resolve_model_path_to_posix(
                llm_model.model_path,
                default_relative_directory="llama_models/ggml",
            ),
            n_ctx=llm_model.max_total_tokens,
            n_parts=llm_model.n_parts,
            n_gpu_layers=llm_model.n_gpu_layers,
            seed=llm_model.seed,
            f16_kv=llm_model.f16_kv,
            logits_all=llm_model.logits_all,
            vocab_only=llm_model.vocab_only,
            use_mmap=llm_model.use_mmap,
            use_mlock=llm_model.use_mlock,
            embedding=llm_model.embedding,
            n_threads=llm_model.n_threads,
            n_batch=llm_model.n_batch,
            last_n_tokens_size=llm_model.last_n_tokens_size,
            lora_base=llm_model.lora_base,
            lora_path=llm_model.lora_path,
            low_vram=llm_model.low_vram,
            verbose=llm_model.echo,
        )
        if llm_model.cache:
            cache_type = "ram" if llm_model.cache_type is None else "disk"
            cache_size = (
                2 << 30 if llm_model.cache_size is None else llm_model.cache_size
            )
            if cache_type == "disk":
                if llm_model.echo:
                    logger.info(
                        f"🦙 Using disk cache with size {cache_size}",
                    )
                cache = llama_cpp.LlamaDiskCache(capacity_bytes=cache_size)
            else:
                if llm_model.echo:
                    logger.info(
                        f"🦙 Using ram cache with size {cache_size}",
                    )
                cache = llama_cpp.LlamaRAMCache(capacity_bytes=cache_size)
            client.set_cache(cache)
        self = cls()
        self.client = client
        self._llm_model = llm_model
        return self

    def generate_completion(
        self,
        prompt: str,
        settings: TextGenerationSettings = TextGenerationSettings(),
    ) -> Completion:
        assert self.client is not None
        completion = _create_completion(
            client=self.client, prompt=prompt, stream=False, settings=settings
        )
        assert not isinstance(completion, Iterator)
        return completion

    def generate_completion_with_streaming(
        self,
        prompt: str,
        settings: TextGenerationSettings = TextGenerationSettings(),
    ) -> Iterator[CompletionChunk]:
        assert self.client is not None
        completion_chunk_generator = _create_completion(
            client=self.client, prompt=prompt, stream=True, settings=settings
        )
        assert isinstance(completion_chunk_generator, Iterator)
        self.generator = completion_chunk_generator
        yield from completion_chunk_generator

    def generate_chat_completion(
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> ChatCompletion:
        assert self.client is not None
        chat_completion = _create_chat_completion(
            client=self.client, messages=messages, stream=False, settings=settings
        )
        assert not isinstance(chat_completion, Iterator)
        return chat_completion

    def generate_chat_completion_with_streaming(
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> Iterator[ChatCompletionChunk]:
        assert self.client is not None
        chat_completion_chunk_generator = _create_chat_completion(
            client=self.client, messages=messages, stream=True, settings=settings
        )
        assert isinstance(chat_completion_chunk_generator, Iterator)
        self.generator = chat_completion_chunk_generator
        yield from chat_completion_chunk_generator


if __name__ == "__main__":
    from app.models.llms import LlamaCppModel
    from app.models.llm_tokenizers import LlamaTokenizer

    llama_cpp_model = LlamaCppModel(
        model_path="orca-mini-3b.ggmlv3.q4_1.bin",
        name="orca_mini_7b",
        max_total_tokens=1024,
        max_tokens_per_request=1024,
        token_margin=8,
        tokenizer=LlamaTokenizer("orca_mini_7b"),
    )
    generator: LlamaCppCompletionGenerator = (
        LlamaCppCompletionGenerator.from_pretrained(llama_cpp_model)
    )
    for completion in generator.generate_completion_with_streaming(
        prompt="Hello, my name is", settings=TextGenerationSettings()
    ):
        print(completion)

    print(
        generator.generate_completion(
            prompt="Hello, my name is", settings=TextGenerationSettings()
        )
    )
