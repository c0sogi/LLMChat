from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import TYPE_CHECKING
from app.utils.chat.text_generations.path import resolve_model_path_to_posix

from app.utils.logger import ApiLogger

if TYPE_CHECKING:
    from tiktoken import Encoding
    from transformers.models.llama import LlamaTokenizer as _LlamaTokenizer
    from repositories.exllama.tokenizer import ExLlamaTokenizer


class BaseTokenizer(ABC):
    @abstractproperty
    def tokenizer(self):
        ...

    @abstractmethod
    def encode(self, message: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        ...

    @abstractmethod
    def tokens_of(self, message: str) -> int:
        ...

    def split_text_on_tokens(
        self, text: str, tokens_per_chunk: int, chunk_overlap: int
    ) -> list[str]:
        """Split incoming text and return chunks."""
        splits: list[str] = []
        input_ids = self.encode(text)
        start_idx = 0
        cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            splits.append(self.decode(chunk_ids))
            start_idx += tokens_per_chunk - chunk_overlap
            cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
        return splits

    def get_chunk_of(self, text: str, tokens: int) -> str:
        """Split incoming text and return chunks."""
        input_ids = self.encode(text)
        return self.decode(input_ids[: min(tokens, len(input_ids))])


class OpenAITokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer: "Encoding" | None = None

    def encode(self, message: str, /) -> list[int]:
        return self.tokenizer.encode(message)

    def decode(self, tokens: list[int], /) -> str:
        return self.tokenizer.decode(tokens)

    def tokens_of(self, message: str) -> int:
        return len(self.encode(message))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def tokenizer(self) -> "Encoding":
        from tiktoken import encoding_for_model

        if self._tokenizer is None:
            print("Loading tokenizer: ", self.model_name)
            self._tokenizer = encoding_for_model(self.model_name)
        return self._tokenizer


class LlamaTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        from transformers.models.llama import LlamaTokenizer as _LlamaTokenizer

        self.model_name = model_name
        self._tokenizer = None
        self._tokenizer_type = _LlamaTokenizer

    def encode(self, message: str, /) -> list[int]:
        return self.tokenizer.encode(message)

    def decode(self, tokens: list[int], /) -> str:
        return self.tokenizer.decode(tokens)

    def tokens_of(self, message: str) -> int:
        return len(self.encode(message))

    @property
    def tokenizer(self) -> "_LlamaTokenizer":
        if self._tokenizer is None:
            split_str = self.model_name.split("/")

            if len(split_str) == 2:
                root_path = self.model_name
                subfolder = None
            elif len(split_str) > 2:
                root_path = "/".join(split_str[:2])
                subfolder = "/".join(split_str[2:])
            else:
                print(split_str)
                raise ValueError("Input string is not in the correct format")
            try:
                self._tokenizer = self._tokenizer_type.from_pretrained(
                    root_path, subfolder=subfolder
                )
            except Exception as e:
                ApiLogger.cerror(
                    f"Error loading tokenizer: {self.model_name}", exc_info=True
                )
                raise e
            print("Tokenizer loaded:", self.model_name)
        return self._tokenizer


class ExllamaTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        from repositories.exllama.tokenizer import ExLlamaTokenizer as _ExllamaTokenizer

        self.model_name = model_name
        self._tokenizer = None
        self._tokenizer_type = _ExllamaTokenizer

    def encode(self, message: str, /) -> list[int]:
        return self.tokenizer.encode(message).flatten().tolist()

    def decode(self, tokens: list[int], /) -> str:
        from torch import IntTensor

        return str(self.tokenizer.decode(IntTensor(tokens)))

    def tokens_of(self, message: str) -> int:
        return len(self.encode(message))

    @property
    def tokenizer(self) -> "ExLlamaTokenizer":
        if self._tokenizer is None:
            model_folder_path = Path(
                resolve_model_path_to_posix(
                    self.model_name,
                    default_relative_directory="llama_models/gptq",
                ),
            )
            try:
                self._tokenizer = self._tokenizer_type(
                    (model_folder_path / "tokenizer.model").as_posix(),
                )
            except Exception as e:
                ApiLogger.cerror(
                    f"Error loading tokenizer: {self.model_name}", exc_info=True
                )
                raise e
            print("Tokenizer loaded:", self.model_name)
        return self._tokenizer
