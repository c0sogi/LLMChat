from gc import collect
from typing import Optional

from torch import Tensor, cuda
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

from . import BaseEmbeddingGenerator

device = "cuda" if cuda.is_available() else "cpu"


class TransformerEmbeddingGenerator(BaseEmbeddingGenerator):
    """Generate embeddings using a transformer model,
    automatically downloading the model from https://huggingface.co/"""

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None
    encoder: Optional[PreTrainedModel] = None
    _model_name: Optional[str] = None

    def __del__(self) -> None:
        if self.model is not None:
            getattr(self.model, "__del__", lambda: None)()
            self.model = None
            print("🗑️ TransformerEmbedding model deleted!")
        if self.tokenizer is not None:
            getattr(self.tokenizer, "__del__", lambda: None)()
            self.tokenizer = None
            print("🗑️ TransformerEmbedding tokenizer deleted!")
        if self.encoder is not None:
            getattr(self.encoder, "__del__", lambda: None)()
            self.encoder = None
            print("🗑️ TransformerEmbedding encoder deleted!")

    @classmethod
    def from_pretrained(
        cls, model_name: str
    ) -> "TransformerEmbeddingGenerator":
        self = cls()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model_name = model_name
        ApiLogger("||embeddings||").info(
            f"🤖 Huggingface tokenizer {Fore.WHITE}{model_name}{Fore.GREEN} loaded!",
        )

        self.model = AutoModel.from_pretrained(model_name)
        ApiLogger("||embeddings||").info(
            f"🤖 Huggingface model {Fore.WHITE}{model_name}{Fore.GREEN} loaded!",
        )
        return self

    def generate_embeddings(
        self,
        texts: list[str],
        context_length: int = 512,
        batch_size: int = 3,
        **kwargs,
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx : (batch_idx + batch_size)]
            batch_embeddings, _ = self._generate_embeddings_and_n_tokens(
                texts=batch_texts, context_length=context_length
            )
            embeddings.extend(batch_embeddings)
        return embeddings

    def _generate_embeddings_and_n_tokens(
        self,
        texts: list[str],
        context_length: int = 512,
    ) -> tuple[list[list[float]], int]:
        assert self.model is not None and self.tokenizer is not None

        def average_pool(
            last_hidden_states: Tensor, attention_mask: Tensor
        ) -> Tensor:
            last_hidden = last_hidden_states.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            return (
                last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            )

        # Tokenize the input texts
        batch_dict: BatchEncoding = self.tokenizer(
            texts,
            max_length=context_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        if self.encoder is None:
            # Get the encoder from the model
            if isinstance(self.model, T5Model):
                self.encoder = self.model.get_encoder()
            else:
                self.encoder = self.model

        if device == "cuda":
            # Load the encoder into VRAM
            self.encoder = self.encoder.to(device)
            batch_dict = batch_dict.to(device)
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
            **batch_dict
        )
        embeddings, tokens = (
            average_pool(
                last_hidden_states=outputs.last_hidden_state,
                attention_mask=batch_dict["attention_mask"],  # type: ignore
            ).tolist(),
            sum([len(encoding) for encoding in batch_dict["input_ids"]]),  # type: ignore
        )
        del batch_dict
        del outputs
        if device == "cuda":
            # Deallocate output tensors from VRAM
            cuda.empty_cache()
        collect()
        return embeddings, tokens

    @property
    def model_name(self) -> str:
        return self._model_name or self.__class__.__name__
