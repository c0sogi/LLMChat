"""Wrapper around the Milvus vector database."""

from hashlib import md5
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.qdrant import MetadataFilter
from langchain.vectorstores.qdrant import Qdrant as _Qdrant
from langchain.vectorstores.utils import maximal_marginal_relevance
import numpy as np

if TYPE_CHECKING or True:
    from qdrant_client import grpc


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
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        from qdrant_client import grpc
        from qdrant_client.conversions.conversion import payload_to_grpc

        grpc_points = self.client.async_grpc_points
        texts = list(texts)  # otherwise iterable might be exhausted after id calculation
        ids = [md5(text.encode("utf-8")).hexdigest() for text in texts]
        vectors = self._embed_texts(texts)
        payloads = self._build_payloads(
            texts=texts,
            metadatas=metadatas,
            content_payload_key=self.content_payload_key,
            metadata_payload_key=self.metadata_payload_key,
        )
        points = [
            grpc.PointStruct(  # type: ignore
                id=grpc.PointId(uuid=id),  # type: ignore
                vectors=grpc.Vectors(vector=grpc.Vector(data=vector)),  # type: ignore
                payload=payload_to_grpc(payload),
            )
            for id, vector, payload in zip(ids, vectors, payloads)  # type: ignore
        ]
        await grpc_points.Upsert(
            grpc.UpsertPoints(  # type: ignore
                collection_name=collection_name if collection_name is not None else self.collection_name,
                points=points,
            )
        )

        return ids

    async def asimilarity_search_with_score(
        self,
        query: str,
        collection_name: Optional[str] = None,
        k: int = 4,
        filter: Optional[MetadataFilter] = None,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        from qdrant_client import grpc

        grpc_points = self.client.async_grpc_points
        if filter is not None:
            search_filter = grpc.Filter(  # type: ignore
                must=[
                    condition for key, value in filter.items() for condition in self._build_condition_grpc(key, value)
                ]
            )
        else:
            search_filter = None
        response = await grpc_points.Search(
            grpc.SearchPoints(  # type: ignore
                collection_name=collection_name if collection_name is not None else self.collection_name,
                vector=self._embed_query(query),
                filter=search_filter,
                with_payload=grpc.WithPayloadSelector(enable=True),  # type: ignore
                limit=k,
            )
        )

        return [
            (
                self._document_from_scored_point_grpc(result, self.content_payload_key, self.metadata_payload_key),
                result.score,
            )
            for result in response.result  # type: ignore
        ]

    async def asimilarity_search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        k: int = 4,
        filter: Optional[MetadataFilter] = None,
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
            collection_name=collection_name if collection_name is not None else self.collection_name,
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
        embedding = self._embed_query(query)

        response = await grpc_points.Search(
            grpc.SearchPoints(  # type: ignore
                collection_name=collection_name if collection_name is not None else self.collection_name,
                vector=embedding,
                with_payload=grpc.WithPayloadSelector(enable=True),  # type: ignore
                with_vectors=grpc.WithVectorsSelector(enable=True),  # type: ignore
                limit=fetch_k,
            )
        )
        # print("scores:", [result.score for result in response.result])
        embeddings: list[rest.VectorStruct] = [GrpcToRest.convert_vectors(result.vectors) for result in response.result]
        mmr_selected: list[int] = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )
        return [
            (
                self._document_from_scored_point_grpc(
                    response.result[i], self.content_payload_key, self.metadata_payload_key
                ),
                response.result[i].score,
            )
            for i in mmr_selected
        ]

    def _build_condition_grpc(self, key: str, value: Any) -> List[grpc.Condition]:
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

    def _qdrant_filter_from_dict_grpc(self, filter: Optional[MetadataFilter]) -> Optional[grpc.Filter]:
        from qdrant_client import grpc

        if not filter:
            return None

        return grpc.Filter(  # type: ignore
            must=[condition for key, value in filter.items() for condition in self._build_condition_grpc(key, value)]
        )

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
