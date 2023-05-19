from abc import ABC, abstractmethod
from tiktoken import Encoding, encoding_for_model
from transformers import LlamaTokenizer as _LlamaTokenizer


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, message: str) -> list[int]:
        ...

    @abstractmethod
    def tokens_of(self, message: str) -> int:
        ...


class OpenAITokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer: Encoding | None = None

    def encode(self, message: str, /) -> list[int]:
        return self.tokenizer.encode(message)

    def tokens_of(self, message: str) -> int:
        return len(self.encode(message))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def tokenizer(self) -> Encoding:
        if self._tokenizer is None:
            print("Loading tokenizer: ", self.model_name)
            self._tokenizer = encoding_for_model(self.model_name)
        return self._tokenizer


class LlamaTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer: Encoding | None = None

    def encode(self, message: str, /) -> list[int]:
        return self.tokenizer.encode(message)

    def tokens_of(self, message: str) -> int:
        return len(self.encode(message))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def tokenizer(self) -> Encoding:
        if self._tokenizer is None:
            print("Loading tokenizer: ", self.model_name)
            self._tokenizer = _LlamaTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
