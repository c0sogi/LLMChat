from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from tiktoken import Encoding, encoding_for_model, get_encoding

from app.utils.chat.text_generations.path import resolve_model_path_to_posix
from app.utils.logger import ApiLogger

if TYPE_CHECKING:
    from transformers.models.llama import LlamaTokenizer as _LlamaTokenizer

    from repositories.exllama.tokenizer import ExLlamaTokenizer


class BaseTokenizer(ABC):
    _fallback_tokenizer: Optional[Encoding] = None

    @classmethod
    @property
    def fallback_tokenizer(cls) -> Encoding:
        ApiLogger.cwarning("Using fallback tokenizer!!!")
        if cls._fallback_tokenizer is None:
            cls._fallback_tokenizer = get_encoding("cl100k_base")
        return cls._fallback_tokenizer

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def encode(self, message: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        ...

    def tokens_of(self, message: str) -> int:
        return len(self.encode(message))

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
        self._model_name = model_name
        self._tokenizer: Encoding | None = None

    def encode(self, message: str, /) -> list[int]:
        return self.tokenizer.encode(message)

    def decode(self, tokens: list[int], /) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def tokenizer(self) -> Encoding:
        if self._tokenizer is None:
            print("Loading tokenizer: ", self._model_name)
            self._tokenizer = encoding_for_model(self._model_name)
        return self._tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name


class LlamaTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        try:
            from transformers.models.llama import (
                LlamaTokenizer as _LlamaTokenizer,
            )

            self._tokenizer_type = _LlamaTokenizer
        except Exception as e:
            ApiLogger.cwarning(str(e))
            self._tokenizer_type = None
        self._model_name = model_name
        self._tokenizer = None

    def encode(self, message: str, /) -> list[int]:
        return self.tokenizer.encode(message)

    def decode(self, tokens: list[int], /) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def tokenizer(self) -> Union["_LlamaTokenizer", Encoding]:
        if self._tokenizer is None:
            try:
                if self._tokenizer_type is None:
                    raise Exception("LlamaTokenizer could not be imported")
                split_str = self._model_name.split("/")

                if len(split_str) == 2:
                    root_path = self._model_name
                    subfolder = None
                elif len(split_str) > 2:
                    root_path = "/".join(split_str[:2])
                    subfolder = "/".join(split_str[2:])
                else:
                    raise Exception(
                        f"Input string {split_str} is not in the correct format"
                    )
                self._tokenizer = self._tokenizer_type.from_pretrained(
                    root_path, subfolder=subfolder
                )
                print("Tokenizer loaded:", self._model_name)
            except Exception as e:
                ApiLogger.cwarning(str(e))
                self._tokenizer = self.fallback_tokenizer
        return self._tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name


class ExllamaTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        try:
            from repositories.exllama.tokenizer import (
                ExLlamaTokenizer as _ExllamaTokenizer,
            )

            self._tokenizer_type = _ExllamaTokenizer
        except Exception:
            self._tokenizer_type = None
        self._model_name = model_name
        self._tokenizer = None

    def encode(self, message: str, /) -> list[int]:
        if isinstance(self.tokenizer, Encoding):
            return self.tokenizer.encode(message)
        return self.tokenizer.encode(message).flatten().tolist()

    def decode(self, tokens: list[int], /) -> str:
        if isinstance(self.tokenizer, Encoding):
            return self.tokenizer.decode(tokens)
        from torch import IntTensor

        return str(self.tokenizer.decode(IntTensor(tokens)))

    @property
    def tokenizer(self) -> Union["ExLlamaTokenizer", Encoding]:
        if self._tokenizer is None:
            try:
                if self._tokenizer_type is None:
                    raise Exception("ExllamaTokenizer could not be imported")
                model_folder_path = Path(
                    resolve_model_path_to_posix(
                        self._model_name,
                        default_relative_directory="llama_models/gptq",
                    ),
                )
                self._tokenizer = self._tokenizer_type(
                    (model_folder_path / "tokenizer.model").as_posix(),
                )
                print("Tokenizer loaded:", self._model_name)
            except Exception as e:
                ApiLogger.cwarning(str(e))
                self._tokenizer = self.fallback_tokenizer
        return self._tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name
