from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import tensorflow_hub as hub

from app.utils.colorama import Fore
from app.utils.logger import ApiLogger

from . import BaseEmbeddingGenerator

if TYPE_CHECKING:
    from tensorflow.python.framework.ops import Tensor


class SentenceEncoderEmbeddingGenerator(BaseEmbeddingGenerator):
    """Generate embeddings using a sentence encoder model,
    automatically downloading the model from https://tfhub.dev/"""

    base_url: str = "https://tfhub.dev/google/"
    model: Optional[Callable[[list[str]], "Tensor"]] = None
    _model_name: Optional[str] = None

    def __del__(self) -> None:
        if self.model is not None:
            getattr(self.model, "__del__", lambda: None)()
            del self.model
            self.model = None
            print("🗑️ SentenceEncoderEmbedding deleted!")

    @classmethod
    def from_pretrained(
        cls, model_name: str
    ) -> "SentenceEncoderEmbeddingGenerator":
        self = cls()
        self._model_name = model_name
        url = f"{self.base_url.rstrip('/')}/{model_name.lstrip('/')}"
        self.model = hub.load(url)  # type: ignore
        ApiLogger("||embeddings||").info(
            f"🤖 TFHub {Fore.WHITE}{model_name}{Fore.GREEN} loaded!",
        )
        return self

    def generate_embeddings(
        self,
        texts: list[str],
        batch_size: int = 100,
        **kwargs,
    ) -> list[list[float]]:
        assert self.model is not None, "Please load the model first."
        embeddings: list["Tensor"] = []
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx : (batch_idx + batch_size)]
            embeddings.append(self.model(batch_texts))
        return np.vstack(embeddings).tolist()

    @property
    def model_name(self) -> str:
        return self._model_name or self.__class__.__name__


# class SemanticSearch:
#     use: Callable[[list[str]], "Tensor"]
#     nn: Optional[NearestNeighbors] = None
#     data: Optional[list[str]] = None
#     embeddings: Optional[np.ndarray] = None

#     def __init__(self):
#         self.use = hub.load(
#             "https://tfhub.dev/google/universal-sentence-encoder/4",
#         )  # type: ignore

#     def __call__(self, text: str, return_data: bool = True):
#         assert (
#             self.nn is not None and self.data is not None
#         ), "Please fit the model first."
#         query_embedding: "Tensor" = self.use([text])
#         neighbors: np.ndarray = self.nn.kneighbors(
#             query_embedding,
#             return_distance=False,
#         )[0]

#         if return_data:
#             return [self.data[i] for i in neighbors]
#         else:
#             return neighbors

#     def query(
#         self,
#         text: str,
#         return_data: bool = True,
#     ):
#         return self(text, return_data=return_data)

#     def fit(self, data: list[str], batch: int = 1000, n_neighbors: int = 5):
#         self.data = data
#         self.embeddings = self._get_text_embedding(data, batch=batch)
#         self.nn = NearestNeighbors(
#             n_neighbors=min(
#                 n_neighbors,
#                 len(self.embeddings),
#             )
#         )
#         self.nn.fit(self.embeddings)

#     def _get_text_embedding(
#         self,
#         texts: list[str],
#         batch: int = 1000,
#     ) -> np.ndarray:
#         embeddings: list["Tensor"] = []
#         for batch_idx in range(0, len(texts), batch):
#             text_batch = texts[batch_idx : (batch_idx + batch)]
#             embeddings.append(self.use(text_batch))
#         return np.vstack(embeddings)
