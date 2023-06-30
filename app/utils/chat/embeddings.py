from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from typing import Optional

from torch import Tensor
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Model
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from app.utils.colorama import Fore
from app.utils.logger import ApiLogger


class BaseEmbedding(ABC):
    @abstractclassmethod
    def from_pretrained(cls, pretrained_name: str) -> "BaseEmbedding":
        ...

    @abstractmethod
    def generate_embeddings_and_n_tokens(
        self,
        input_texts: list[str],
        context_length: int,
    ) -> tuple[list[list[float]], int]:
        ...

    @abstractproperty
    def model_name(self) -> str:
        ...


class TransformerEmbedding(BaseEmbedding):
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None
    _model_name: Optional[str] = None

    def __del__(self) -> None:
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        print("🗑️ TransformerEmbedding deleted!")

    @classmethod
    def from_pretrained(cls, pretrained_name: str) -> "TransformerEmbedding":
        self = cls()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self._model_name = pretrained_name
        ApiLogger("||embeddings||").info(
            f"🤖 Huggingface tokenizer {Fore.WHITE}{pretrained_name}{Fore.GREEN} loaded!",
        )

        self.model = AutoModel.from_pretrained(pretrained_name)
        ApiLogger("||embeddings||").info(
            f"🤖 Huggingface model {Fore.WHITE}{pretrained_name}{Fore.GREEN} loaded!",
        )
        return self

    def generate_embeddings_and_n_tokens(
        self,
        input_texts: list[str],
        context_length: int = 512,
    ) -> tuple[list[list[float]], int]:
        assert self.model is not None and self.tokenizer is not None

        def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            last_hidden = last_hidden_states.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        # Tokenize the input texts
        batch_dict: BatchEncoding = self.tokenizer(
            input_texts,
            max_length=context_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        if isinstance(self.model, T5Model):
            encoder = self.model.get_encoder()
        else:
            encoder = self.model
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = encoder(**batch_dict)
        return (
            average_pool(
                last_hidden_states=outputs.last_hidden_state,
                attention_mask=batch_dict["attention_mask"],  # type: ignore
            ).tolist(),
            sum([len(encoding) for encoding in batch_dict["input_ids"]]),  # type: ignore
        )

    @property
    def model_name(self) -> str:
        assert self._model_name is not None
        return self._model_name
