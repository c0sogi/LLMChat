"""Wrapper around the Milvus vector database."""

import warnings
from asyncio import gather
from hashlib import md5
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from fastapi.concurrency import run_in_threadpool
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.qdrant import Qdrant as _Qdrant
from langchain.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from qdrant_client import grpc
    from qdrant_client.conversions import common_types

    DictFilter = Dict[str, Union[str, int, bool, dict, list]]
    MetadataFilter = Union[DictFilter, common_types.Filter]


class Qdrant(_Qdrant):
    """Wrapper around Qdrant vector database.

    To use you should have the ``qdrant-client`` package installed.

    Example:
        .. code-block:: python

            from qdrant_client import QdrantClient
            from langchain import Qdrant

            client = QdrantClient()
            collection_name = "MyCollection"
            qdrant = Qdrant(client, collection_name, embedding_function)
    """

    CONTENT_KEY = "page_content"
    METADATA_KEY = "metadata"

    def __init__(
        self,
        client: Any,
        collection_name: str = "Shared",
        embeddings: Optional[Embeddings] = None,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        embedding_function: Optional[Callable] = None,  # deprecated
    ):
        """Initialize with necessary components."""
        super().__init__(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            embedding_function=embedding_function,
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        collection_name: Optional[str] = None,
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        from itertools import islice

        from qdrant_client import grpc
        from qdrant_client.conversions.conversion import payload_to_grpc

        grpc_points = self.client.async_grpc_points

        ids = []
        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        while batch_texts := list(islice(texts_iterator, batch_size)):
            # Take the corresponding metadata for each text in a batch
            batch_metadatas = (
                list(islice(metadatas_iterator, batch_size)) or None
            )

            batch_ids = [
                md5(text.encode("utf-8")).hexdigest() for text in batch_texts
            ]
            points = [
                grpc.PointStruct(  # type: ignore
                    id=grpc.PointId(uuid=id),  # type: ignore
                    vectors=grpc.Vectors(vector=grpc.Vector(data=vector)),  # type: ignore
                    payload=payload_to_grpc(payload),
                )
                for id, vector, payload in zip(
                    batch_ids,
                    await self._aembed_texts(batch_texts),
                    self._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                )
            ]
            await grpc_points.Upsert(
                grpc.UpsertPoints(  # type: ignore
                    collection_name=collection_name
                    if collection_name is not None
                    else self.collection_name,
                    points=points,
                )
            )
            ids.extend(batch_ids)

        return ids

    async def asimilarity_search_with_score(
        self,
        query: str,
        collection_name: str,
        k: int = 4,
        filter: Optional["MetadataFilter"] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each.
        """

        from qdrant_client.qdrant_remote import QdrantRemote

        if not isinstance(self.client._client, QdrantRemote):
            raise NotImplementedError(
                "Async similarity search is only supported for remote clients",
            )

        from qdrant_client import grpc
        from qdrant_client.conversions.conversion import RestToGrpc
        from qdrant_client.http import models as rest

        grpc_points = self.client.async_grpc_points
        if filter is not None and isinstance(filter, dict):
            warnings.warn(
                "Using dict as a `filter` is deprecated. Please use qdrant-client "
                "filters directly: "
                "https://qdrant.tech/documentation/concepts/filtering/",
                DeprecationWarning,
            )
            qdrant_filter = self._qdrant_filter_from_dict_grpc(filter)
        elif filter is not None and isinstance(filter, rest.Filter):
            qdrant_filter = RestToGrpc.convert_filter(filter)
        else:
            qdrant_filter = filter
        response = await grpc_points.Search(
            grpc.SearchPoints(  # type: ignore
                collection_name=self.collection_name
                if collection_name is None
                else collection_name,
                vector=await self._aembed_query(query),
                filter=qdrant_filter,
                with_payload=grpc.WithPayloadSelector(enable=True),  # type: ignore
                limit=k,
            )
        )

        return [
            (
                self._document_from_scored_point_grpc(
                    result,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                result.score,
            )
            for result in response.result  # type: ignore
        ]

    async def asimilarity_search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        k: int = 4,
        filter: Optional["MetadataFilter"] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        results = await self.asimilarity_search_with_score(
            query=query,
            collection_name=collection_name
            if collection_name is not None
            else self.collection_name,
            k=k,
            filter=filter,
        )
        return list(map(itemgetter(0), results))

    async def amax_marginal_relevance_search_with_score(
        self,
        query: str,
        collection_name: Optional[str] = None,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        from qdrant_client import grpc
        from qdrant_client.conversions.conversion import GrpcToRest
        from qdrant_client.http.models import models as rest

        grpc_points = self.client.async_grpc_points
        embedding = await self._aembed_query(query)

        response = await grpc_points.Search(
            grpc.SearchPoints(  # type: ignore
                collection_name=collection_name
                if collection_name is not None
                else self.collection_name,
                vector=embedding,
                with_payload=grpc.WithPayloadSelector(enable=True),  # type: ignore
                with_vectors=grpc.WithVectorsSelector(enable=True),  # type: ignore
                limit=fetch_k,
            )
        )
        embeddings: list[rest.VectorStruct] = [
            GrpcToRest.convert_vectors(result.vectors)
            for result in response.result
        ]
        mmr_selected: list[int] = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )
        return [
            (
                self._document_from_scored_point_grpc(
                    response.result[i],
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                response.result[i].score,
            )
            for i in mmr_selected
        ]

    def _build_condition_grpc(
        self, key: str, value: Any
    ) -> List["grpc.Condition"]:
        from qdrant_client import grpc

        out: List[grpc.Condition] = []

        if isinstance(value, dict):
            for _key, value in value.items():
                out.extend(self._build_condition_grpc(f"{key}.{_key}", value))
        elif isinstance(value, list):
            for _value in value:
                if isinstance(_value, dict):
                    out.extend(self._build_condition_grpc(f"{key}[]", _value))
                else:
                    out.extend(self._build_condition_grpc(f"{key}", _value))
        else:
            if isinstance(value, str):
                value_type = "text"
            elif isinstance(value, int):
                value_type = "interger"
            elif isinstance(value, bool):
                value_type = "boolean"
            else:
                raise TypeError(f"Unsupported type {type(value)}")

            out.append(
                grpc.Condition(
                    field=grpc.FieldCondition(  # type: ignore
                        key=f"{self.metadata_payload_key}.{key}",
                        match=grpc.Match(**{value_type: value}),  # type: ignore
                    )
                )
            )

        return out

    def _qdrant_filter_from_dict_grpc(
        self, filter: Optional["DictFilter"]
    ) -> Optional["grpc.Filter"]:
        from qdrant_client import grpc

        if not filter:
            return None

        return grpc.Filter(  # type: ignore
            must=[
                condition
                for key, value in filter.items()
                for condition in self._build_condition_grpc(key, value)
            ]
        )

    async def _aembed_query(self, query: str) -> List[float]:
        """Embed query text.

        Used to provide backward compatibility with `embedding_function` argument.

        Args:
            query: Query text.

        Returns:
            List of floats representing the query embedding.
        """
        if self.embeddings is not None:
            embedding = await run_in_threadpool(
                self.embeddings.embed_query, query
            )
        else:
            if self._embeddings_function is not None:
                embedding = await run_in_threadpool(
                    self._embeddings_function, query
                )
            else:
                raise ValueError(
                    "Neither of embeddings or embedding_function is set"
                )
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding  # type: ignore

    async def _aembed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed search texts.

        Used to provide backward compatibility with `embedding_function` argument.

        Args:
            texts: Iterable of texts to embed.

        Returns:
            List of floats representing the texts embedding.
        """
        if self.embeddings is not None:
            embeddings = await run_in_threadpool(
                self.embeddings.embed_documents, list(texts)
            )
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()  # type: ignore
        elif self._embeddings_function is not None:
            embeddings = await gather(
                *[
                    run_in_threadpool(self._embeddings_function, text)
                    for text in texts
                ]
            )

            for embedding_idx in range(len(embeddings)):
                if hasattr(embeddings[embedding_idx], "tolist"):
                    embeddings[embedding_idx] = embeddings[
                        embedding_idx
                    ].tolist()
        else:
            raise ValueError(
                "Neither of embeddings or embedding_function is set"
            )

        return embeddings

    @staticmethod
    def _document_from_scored_point_grpc(
        scored_point: Any,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        from qdrant_client.conversions.conversion import grpc_to_payload

        payload = grpc_to_payload(scored_point.payload)
        return Document(
            page_content=payload[content_payload_key],
            metadata=payload.get(metadata_payload_key) or {},
        )
