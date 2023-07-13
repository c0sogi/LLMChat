"""Wrapper around Redis vector database."""
from __future__ import annotations

from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)
from uuid import uuid4

import numpy as np
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, root_validator

from app.utils.logger import ApiLogger

try:
    from starlette.concurrency import run_in_threadpool
except ImportError:
    raise ImportError(
        "Please install starlette to use the Redis vector store. "
        "pip install starlette"
    )
try:
    import redis

    if TYPE_CHECKING:
        from redis.client import Pipeline as PipelineType
        from redis.client import Redis as RedisType
    from redis.commands.search.field import TextField, VectorField
    from redis.commands.search.indexDefinition import (
        IndexDefinition,
        IndexType,
    )
    from redis.commands.search.query import Query

    import redis.asyncio as aioredis

    if TYPE_CHECKING:
        from redis.asyncio.client import Pipeline as AsyncPipelineType
        from redis.asyncio.client import Redis as AsyncRedisType
except ImportError:
    raise ValueError(
        "Could not import redis python package. "
        "Please install it with `pip install redis`."
    )

# required modules
REDIS_REQUIRED_MODULES = [
    {"name": "search", "ver": 20400},
]


class DistanceMetric(str, Enum):
    """Enum for distance metrics for vectors."""

    EUCLIDEAN = "EUCLIDEAN"
    HAMMING = "HAMMING"
    JACCARD = "JACCARD"
    COSINE = "COSINE"
    DOT_PRODUCT = "DOT_PRODUCT"


def _check_redis_module_exist(client: RedisType, modules: List[dict]) -> None:
    """Check if the correct Redis modules are installed."""
    installed_modules = client.module_list()
    installed_modules = {
        module[b"name"].decode("utf-8"): module for module in installed_modules
    }
    for module in modules:
        if module["name"] not in installed_modules or int(
            installed_modules[module["name"]][b"ver"]
        ) < int(module["ver"]):
            error_message = (
                "You must add the RediSearch (>= 2.4) module from Redis Stack. "
                "Please refer to Redis Stack docs: https://redis.io/docs/stack/"
            )
            ApiLogger.cerror(error_message)
            raise ValueError(error_message)


async def _acheck_redis_module_exist(
    client: AsyncRedisType, modules: List[dict]
) -> None:
    """Check if the correct Redis modules are installed, asynchronously."""
    installed_modules = await client.module_list()
    installed_modules = {
        module[b"name"].decode("utf-8"): module for module in installed_modules
    }
    for module in modules:
        if module["name"] not in installed_modules or int(
            installed_modules[module["name"]][b"ver"]
        ) < int(module["ver"]):
            error_message = (
                "You must add the RediSearch (>= 2.4) module from Redis Stack. "
                "Please refer to Redis Stack docs: https://redis.io/docs/stack/"
            )
            ApiLogger.cerror(error_message)
            raise ValueError(error_message)


def _check_index_exists(client: RedisType, index_name: str) -> bool:
    """Check if Redis index exists."""
    try:
        client.ft(index_name).info()
    except Exception:
        ApiLogger.cinfo("Index does not exist")
        return False
    ApiLogger.cinfo("Index already exists")
    return True


async def _acheck_index_exists(
    client: AsyncRedisType, index_name: str
) -> bool:
    """Check if Redis index exists, asynchronously."""
    try:
        await client.ft(index_name).info()
    except Exception:
        ApiLogger.cinfo("Index does not exist")
        return False
    ApiLogger.cinfo("Index exists!")
    return True


def _redis_client_from_url(redis_url: str, **kwargs: Any) -> RedisType:
    """Create a Redis client from a URL checking if the required modules are installed."""
    try:
        client = redis.from_url(url=redis_url, **kwargs)
        # check if redis has redisearch module installed
        _check_redis_module_exist(client, REDIS_REQUIRED_MODULES)
    except ValueError as e:
        raise ValueError(f"Redis failed to connect: {e}")
    return client


async def _aredis_client_from_url(
    redis_url: str, **kwargs: Any
) -> AsyncRedisType:
    """Create a Redis client from a URL checking if the required modules are installed, asynchronously."""
    try:
        client = aioredis.from_url(url=redis_url, **kwargs)
        # check if redis has redisearch module installed
        await _acheck_redis_module_exist(client, REDIS_REQUIRED_MODULES)
    except ValueError as e:
        raise ValueError(f"Redis failed to connect: {e}")
    return client


def _redis_key(prefix: str) -> str:
    """Redis key schema for a given prefix."""
    return f"{prefix}:{uuid4().hex}"


def _redis_prefix(index_name: str) -> str:
    """Redis key prefix for a given index."""
    return f"doc:{index_name}"


def _redis_vectorstore_schema(
    content_key: str,
    metadata_key: str,
    vector_key: str,
    dim: int,
    distance_metric: str,
) -> Tuple[TextField, TextField, VectorField]:
    return (
        TextField(name=content_key),
        TextField(name=metadata_key),
        VectorField(
            vector_key,
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": distance_metric,
            },
        ),
    )


def _redis_embed_texts_to_pipeline(
    texts: List[str],
    prefix: str,
    content_key: str,
    metadata_key: str,
    vector_key: str,
    embeddings: List[List[float]],
    pipeline: Union[PipelineType, AsyncPipelineType],
    metadatas: Optional[List[dict]] = None,
) -> None:
    for i, text in enumerate(texts):
        key = _redis_key(prefix)
        metadata = metadatas[i] if metadatas else {}
        pipeline.hset(
            key,
            mapping={
                content_key: text,
                vector_key: np.array(
                    embeddings[i], dtype=np.float32
                ).tobytes(),
                metadata_key: orjson_dumps(metadata),
            },
        )


def _ensure_index_exist(
    client: RedisType,
    index_name: str,
    prefix: str,
    content_key: str,
    metadata_key: str,
    vector_key: str,
    dim: int,
) -> None:
    if not _check_index_exists(client, index_name):
        # Constants
        schema = _redis_vectorstore_schema(
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            dim=dim,
            distance_metric=DistanceMetric.COSINE.value,
        )
        # Create Redis Index
        client.ft(index_name).create_index(
            fields=schema,
            definition=IndexDefinition(
                prefix=[prefix], index_type=IndexType.HASH
            ),
        )


async def _aensure_index_exist(
    client: AsyncRedisType,
    index_name: str,
    prefix: str,
    dim: int,
    content_key: str = "content",
    metadata_key: str = "metadata",
    vector_key: str = "content_vector",
) -> None:
    if not await _acheck_index_exists(client, index_name):
        # Constants
        schema = _redis_vectorstore_schema(
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            dim=dim,
            distance_metric=DistanceMetric.COSINE.value,
        )
        # Create Redis Index
        await client.ft(index_name).create_index(
            fields=schema,
            definition=IndexDefinition(
                prefix=[prefix], index_type=IndexType.HASH
            ),
        )


class Redis(VectorStore):
    def __init__(
        self,
        redis_url: str,
        embedding_function: Callable,
        content_key: str = "content",
        metadata_key: str = "metadata",
        vector_key: str = "content_vector",
        is_async: bool = False,
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        # We need to first remove redis_url from kwargs,
        # otherwise passing it to Redis will result in an error.
        if "redis_url" in kwargs:
            kwargs.pop("redis_url")
        client = _redis_client_from_url(redis_url=redis_url, **kwargs)
        if is_async:
            client.close()
            self.client = aioredis.from_url(url=redis_url, **kwargs)
        else:
            self.client = client  # type: ignore
        self.content_key = content_key
        self.metadata_key = metadata_key
        self.vector_key = vector_key

    def _add_texts(
        self,
        texts: Iterable[str],
        index_name: str,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Tuple[List[str], Union[PipelineType, AsyncPipelineType]]:
        prefix = _redis_prefix(index_name)
        keys = kwargs.get("keys")
        ids = []
        # Write data to redis
        pipeline = self.client.pipeline(transaction=False)
        for i, text in enumerate(texts):
            # Use provided key otherwise use default key
            key = keys[i] if keys else _redis_key(prefix)
            metadata = metadatas[i] if metadatas else {}
            pipeline.hset(
                key,
                mapping={
                    self.content_key: text,
                    self.vector_key: np.array(
                        self.embedding_function(text), dtype=np.float32
                    ).tobytes(),
                    self.metadata_key: orjson_dumps(metadata),
                },
            )
            ids.append(key)
        return ids, pipeline

    def add_texts(
        self,
        texts: Iterable[str],
        index_name: str,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts data to an existing index."""
        ids, pipeline = self._add_texts(
            texts, index_name=index_name, metadatas=metadatas, **kwargs
        )
        pipeline.execute()
        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        index_name: str,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts data to an existing index."""
        ids, pipeline = await run_in_threadpool(
            self._add_texts, texts, index_name, metadatas, **kwargs
        )
        await pipeline.execute()  # type: ignore
        return ids

    def similarity_search(
        self, query: str, index_name: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, index_name=index_name, k=k
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search(
        self, query: str, index_name: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text, asynchronously.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = await self.asimilarity_search_with_score(
            query, index_name=index_name, k=k
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_limit_score(
        self,
        query: str,
        index_name: str,
        k: int = 4,
        score_threshold: float = 0.2,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text within the
        score_threshold range.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            score_threshold (float): The minimum matching score required for a document
            to be considered a match. Defaults to 0.2.
            Because the similarity calculation algorithm is based on cosine similarity,
            the smaller the angle, the higher the similarity.

        Returns:
            List[Document]: A list of documents that are most similar to the query text,
            including the match score for each document.

        Note:
            If there are no documents that satisfy the score_threshold value,
            an empty list is returned.

        """
        docs_and_scores = self.similarity_search_with_score(
            query, index_name=index_name, k=k
        )
        return [
            doc for doc, score in docs_and_scores if score < score_threshold
        ]

    async def asimilarity_search_limit_score(
        self,
        query: str,
        index_name: str,
        k: int = 4,
        score_threshold: float = 0.2,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text within the
        score_threshold range, asynchronously.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            score_threshold (float): The minimum matching score required for a document
            to be considered a match. Defaults to 0.2.
            Because the similarity calculation algorithm is based on cosine similarity,
            the smaller the angle, the higher the similarity.

        Returns:
            List[Document]: A list of documents that are most similar to the query text,
            including the match score for each document.

        Note:
            If there are no documents that satisfy the score_threshold value,
            an empty list is returned.

        """
        docs_and_scores = await self.asimilarity_search_with_score(
            query, index_name=index_name, k=k
        )
        return [
            doc for doc, score in docs_and_scores if score < score_threshold
        ]

    def _similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> Tuple[Query, Mapping[str, str]]:
        # Creates embedding vector from user query
        embedding = self.embedding_function(query)

        # Prepare the Query
        return_fields = [self.metadata_key, self.content_key, "vector_score"]
        vector_field = self.vector_key
        hybrid_fields = "*"
        base_query = f"{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]"
        redis_query = (
            Query(base_query)
            .return_fields(*return_fields)
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )
        params_dict: Mapping[str, str] = {
            "vector": np.array(embedding).astype(dtype=np.float32).tobytes()  # type: ignore
        }
        return redis_query, params_dict

    def similarity_search_with_score(
        self, query: str, index_name: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        redis_query, params_dict = self._similarity_search_with_score(
            query, k=k
        )

        # perform vector search
        results = self.client.ft(index_name).search(redis_query, params_dict)

        docs = [
            (
                Document(
                    page_content=result.content,
                    metadata=orjson_loads(result.metadata),
                ),
                float(result.vector_score),
            )
            for result in results.docs
        ]

        return docs

    async def asimilarity_search_with_score(
        self, query: str, index_name: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query, asynchronously.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        redis_query, params_dict = await run_in_threadpool(
            self._similarity_search_with_score, query, k
        )

        # perform vector search
        results = await self.client.ft(index_name).search(redis_query, params_dict)  # type: ignore

        docs = [
            (
                Document(
                    page_content=result.content,
                    metadata=orjson_loads(result.metadata),
                ),
                float(result.vector_score),
            )
            for result in results.docs
        ]
        return docs

    @classmethod
    def from_texts(
        cls: Type[Redis],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        content_key: str = "content",
        metadata_key: str = "metadata",
        vector_key: str = "content_vector",
        **kwargs: Any,
    ) -> Redis:
        """Construct RediSearch wrapper from raw documents.
        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the RediSearch instance.
            3. Adds the documents to the newly created RediSearch index.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python
                from langchain import RediSearch
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redisearch = RediSearch.from_texts(
                    texts,
                    embeddings,
                    redis_url="redis://username:password@localhost:6379"
                )
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        # We need to first remove redis_url from kwargs,
        # otherwise passing it to Redis will result in an error.
        if "redis_url" in kwargs:
            kwargs.pop("redis_url")
        client = _redis_client_from_url(redis_url=redis_url, **kwargs)

        # Create embeddings over documents
        embeddings = embedding.embed_documents(texts)

        # Name of the search index if not given
        if not index_name:
            index_name = uuid4().hex
        prefix = _redis_prefix(index_name)  # prefix for the document keys

        # Check if index exists
        _ensure_index_exist(
            client=client,
            index_name=index_name,
            prefix=prefix,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            dim=len(embeddings[0]),
        )

        # Write data to Redis
        pipeline = client.pipeline(transaction=False)
        _redis_embed_texts_to_pipeline(
            pipeline=pipeline,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            prefix=prefix,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
        )
        pipeline.execute()
        return cls(
            redis_url,
            embedding.embed_query,
            index_name=index_name,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            is_async=False,
            **kwargs,
        )

    @classmethod
    async def afrom_texts(
        cls: Type[Redis],
        texts: List[str],
        embedding: Embeddings,
        index_name: str,
        metadatas: Optional[List[dict]] = None,
        content_key: str = "content",
        metadata_key: str = "metadata",
        vector_key: str = "content_vector",
        **kwargs: Any,
    ) -> Redis:
        """Construct RediSearch wrapper from raw documents, asynchronously.
        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the RediSearch instance.
            3. Adds the documents to the newly created RediSearch index.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python
                from langchain import RediSearch
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redisearch = await RediSearch.afrom_texts(
                    texts,
                    embeddings,
                    redis_url="redis://username:password@localhost:6379"
                )
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        # We need to first remove redis_url from kwargs,
        # otherwise passing it to Redis will result in an error.
        if "redis_url" in kwargs:
            kwargs.pop("redis_url")
        client = await _aredis_client_from_url(redis_url=redis_url, **kwargs)

        # Create embeddings over documents
        embeddings = embedding.embed_documents(texts)
        prefix = _redis_prefix(index_name)  # prefix for the document keys

        # Check if index exists
        await _aensure_index_exist(
            client=client,
            index_name=index_name,
            prefix=prefix,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            dim=len(embeddings[0]),
        )

        # Write data to Redis
        pipeline = client.pipeline(transaction=False)
        _redis_embed_texts_to_pipeline(
            pipeline=pipeline,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            prefix=prefix,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
        )
        await pipeline.execute()
        return cls(
            redis_url,
            embedding.embed_query,
            index_name=index_name,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            is_async=True,
            **kwargs,
        )

    @staticmethod
    def drop_index(
        index_name: str,
        delete_documents: bool,
        **kwargs: Any,
    ) -> bool:
        """
        Drop a Redis search index.

        Args:
            index_name (str): Name of the index to drop.
            delete_documents (bool): Whether to drop the associated documents.

        Returns:
            bool: Whether or not the drop was successful.
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        # We need to first remove redis_url from kwargs,
        # otherwise passing it to Redis will result in an error.
        if "redis_url" in kwargs:
            kwargs.pop("redis_url")
        client = _redis_client_from_url(redis_url=redis_url, **kwargs)
        # Check if index exists
        try:
            client.ft(index_name).dropindex(delete_documents)
            ApiLogger.cinfo("Drop index")
            return True
        except Exception:
            # Index not exist
            return False

    @staticmethod
    async def adrop_index(
        index_name: str,
        delete_documents: bool,
        **kwargs: Any,
    ) -> bool:
        """
        Drop a Redis search index, asynchronously.

        Args:
            index_name (str): Name of the index to drop.
            delete_documents (bool): Whether to drop the associated documents.

        Returns:
            bool: Whether or not the drop was successful.
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        # We need to first remove redis_url from kwargs,
        # otherwise passing it to Redis will result in an error.
        if "redis_url" in kwargs:
            kwargs.pop("redis_url")
        try:
            client = await _aredis_client_from_url(
                redis_url=redis_url, **kwargs
            )
        except ValueError as e:
            raise ValueError(f"Your redis connected error: {e}")
        # Check if index exists
        try:
            await client.ft(index_name).dropindex(delete_documents)
            ApiLogger.cinfo("Drop index")
            return True
        except Exception:
            # Index not exist
            return False

    @classmethod
    def from_existing_index(
        cls,
        embedding: Embeddings,
        index_name: str,
        content_key: str = "content",
        metadata_key: str = "metadata",
        vector_key: str = "content_vector",
        **kwargs: Any,
    ) -> Redis:
        """Connect to an existing Redis index."""
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        if "redis_url" in kwargs:
            kwargs.pop("redis_url")
        try:
            client = _redis_client_from_url(redis_url=redis_url, **kwargs)
            # ensure that the index already exists
            assert _check_index_exists(
                client, index_name
            ), f"Index {index_name} does not exist"
        except Exception as e:
            raise ValueError(f"Redis failed to connect: {e}")

        return cls(
            redis_url,
            embedding.embed_query,
            index_name=index_name,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            is_async=False,
            **kwargs,
        )

    @classmethod
    async def afrom_existing_index(
        cls,
        embedding: Embeddings,
        index_name: str,
        content_key: str = "content",
        metadata_key: str = "metadata",
        vector_key: str = "content_vector",
        **kwargs: Any,
    ) -> Redis:
        """Connect to an existing Redis index, asynchronously."""
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        if "redis_url" in kwargs:
            kwargs.pop("redis_url")
        try:
            client = await _aredis_client_from_url(
                redis_url=redis_url, **kwargs
            )
            # ensure that the index already exists
            assert await _acheck_index_exists(
                client, index_name
            ), f"Index {index_name} does not exist"
        except Exception as e:
            raise ValueError(f"Redis failed to connect: {e}")

        return cls(
            redis_url,
            embedding.embed_query,
            index_name=index_name,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            is_async=True,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        return RedisVectorStoreRetriever(vectorstore=self, **kwargs)


class RedisVectorStoreRetriever(BaseRetriever, BaseModel):
    vectorstore: Redis
    search_type: str = "similarity"
    k: int = 4
    score_threshold: float = 0.4

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in ("similarity", "similarity_limit"):
                raise ValueError(f"search_type of {search_type} not allowed.")
        return values

    def _get_relevant_documents(
        self, query: str, index_name: str
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(
                query, index_name=index_name, k=self.k
            )
        elif self.search_type == "similarity_limit":
            docs = self.vectorstore.similarity_search_limit_score(
                query,
                index_name=index_name,
                k=self.k,
                score_threshold=self.score_threshold,
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self, query: str, index_name: str
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(
                query, index_name=index_name, k=self.k
            )
        elif self.search_type == "similarity_limit":
            docs = await self.vectorstore.asimilarity_search_limit_score(
                query,
                index_name=index_name,
                k=self.k,
                score_threshold=self.score_threshold,
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs
