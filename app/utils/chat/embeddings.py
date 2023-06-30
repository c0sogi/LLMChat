from collections import deque
from gc import collect
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


class TransformerEmbedding:
    def __del__(self) -> None:
        del self.model
        del self.tokenizer
        collect()

    pretrained_name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast


transformer_embeddings: deque[TransformerEmbedding] = deque(maxlen=1)


def _find_transformer_embedding_by_name(name: str) -> Optional[TransformerEmbedding]:
    for embedding in transformer_embeddings:
        if embedding.pretrained_name == name:
            return embedding


def get_embeddings_and_num_of_tokens(
    pretrained_name: str,
    input_texts: list[str],
    context_length: int = 512,
) -> tuple[list[list[float]], int]:
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    transformer_embedding: Optional[
        TransformerEmbedding
    ] = _find_transformer_embedding_by_name(pretrained_name)
    if transformer_embedding is None:
        transformer_embedding = TransformerEmbedding()
        transformer_embedding.pretrained_name = pretrained_name
        transformer_embedding.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        ApiLogger("||embeddings||").info(
            f"🤖 Huggingface tokenizer {Fore.WHITE}{pretrained_name}{Fore.GREEN} loaded!",
        )

        transformer_embedding.model = AutoModel.from_pretrained(pretrained_name)
        ApiLogger("||embeddings||").info(
            f"🤖 Huggingface model {Fore.WHITE}{pretrained_name}{Fore.GREEN} loaded!",
        )

    # Tokenize the input texts
    batch_dict: BatchEncoding = transformer_embedding.tokenizer(
        input_texts,
        max_length=context_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    if isinstance(transformer_embedding.model, T5Model):
        encoder = transformer_embedding.model.get_encoder()
    else:
        encoder = transformer_embedding.model
    outputs: BaseModelOutputWithPoolingAndCrossAttentions = encoder(**batch_dict)
    return (
        average_pool(
            last_hidden_states=outputs.last_hidden_state,
            attention_mask=batch_dict["attention_mask"],  # type: ignore
        ).tolist(),
        sum([len(encoding) for encoding in batch_dict["input_ids"]]),  # type: ignore
    )
