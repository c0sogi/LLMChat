from asyncio import gather
from typing import Any, Coroutine

from fastapi.concurrency import run_in_threadpool
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from qdrant_client import QdrantClient

from app.common.config import (
    EMBEDDING_TOKEN_CHUNK_OVERLAP,
    EMBEDDING_TOKEN_CHUNK_SIZE,
)
from app.common.lotties import Lotties
from app.database.connection import cache
from app.utils.chat.file_loader import read_bytes_to_text


class VectorStoreManager:
    @staticmethod
    async def get_vector_size() -> int:
        return len(await cache.vectorstore._aembed_query("foo"))

    @staticmethod
    async def create_documents(
        text: str,
        collection_name: str,
        chunk_size: int = EMBEDDING_TOKEN_CHUNK_SIZE,
        chunk_overlap: int = EMBEDDING_TOKEN_CHUNK_OVERLAP,
        tokenizer_model: str = "gpt-3.5-turbo",
    ) -> list[str]:
        texts = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=tokenizer_model,
        ).split_text(text)
        assert isinstance(cache.vectorstore.client, QdrantClient)
        if (
            collection_name
            not in await VectorStoreManager.get_all_collection_names()
        ):
            await VectorStoreManager.create_collection(
                collection_name=collection_name,
            )

        await cache.vectorstore.aadd_texts(
            texts, collection_name=collection_name
        )
        return texts

    @staticmethod
    async def asimilarity_search(
        query: str,
        collection_name: str,
        k: int = 1,
    ) -> list[Document]:
        if (
            collection_name
            not in await VectorStoreManager.get_all_collection_names()
        ):
            return []
        return await cache.vectorstore.asimilarity_search(
            query, collection_name=collection_name, k=k
        )

    @staticmethod
    async def asimilarity_search_multiple_collections(
        query: str,
        collection_names: list[str],
        k: int = 1,
    ) -> list[Document]:
        # Nested function to handle index check and search.
        # results: list[list[Document]] =   # shape: (index, k)
        # Reorganize results to have shape: (index * k)

        query_tasks: list[Coroutine[Any, Any, list[Document]]] = []
        existing_collection_names: list[
            str
        ] = await VectorStoreManager.get_all_collection_names()
        for collection_name in collection_names:
            if collection_name in existing_collection_names:
                query_tasks.append(
                    cache.vectorstore.asimilarity_search(
                        query,
                        collection_name=collection_name,
                        k=k,
                    )
                )
        return [
            item for sublist in await gather(*query_tasks) for item in sublist
        ]

    @staticmethod
    async def asimilarity_search_multiple_collections_with_score(
        query: str,
        collection_names: list[str],
        k: int = 1,
    ) -> list[tuple[Document, float]]:
        # Nested function to handle index check and search.
        # search_with_score_if_index_exists's shape: k
        # search_with_score_if_index_exists's return type: list[tuple[Document, float]]
        # Assume that tuple[Document, float] is a single result.
        # Reorganize results to have shape: (index * k), sorted by score.

        query_tasks: list[
            Coroutine[Any, Any, list[tuple[Document, float]]]
        ] = []
        existing_collection_names: list[
            str
        ] = await VectorStoreManager.get_all_collection_names()
        for collection_name in collection_names:
            if collection_name in existing_collection_names:
                query_tasks.append(
                    cache.vectorstore.asimilarity_search_with_score(
                        query,
                        collection_name=collection_name,
                        k=k,
                    )
                )
        return sorted(
            [
                item
                for sublist in await gather(*query_tasks)
                for item in sublist
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    @staticmethod
    async def amax_marginal_relevance_search_multiple_collections_with_score(
        query: str,
        collection_names: list[str],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        # Nested function to handle index check and search.
        # search_with_score_if_index_exists's shape: k
        # search_with_score_if_index_exists's return type: list[tuple[Document, float]]
        # Assume that tuple[Document, float] is a single result.
        # Reorganize results to have shape: (index * k), sorted by score.

        query_tasks: list[
            Coroutine[Any, Any, list[tuple[Document, float]]]
        ] = []
        existing_collection_names: list[
            str
        ] = await VectorStoreManager.get_all_collection_names()
        for collection_name in collection_names:
            if collection_name in existing_collection_names:
                query_tasks.append(
                    cache.vectorstore.amax_marginal_relevance_search_with_score(
                        query=query,
                        collection_name=collection_name,
                        k=k,
                        fetch_k=fetch_k,
                        lambda_mult=lambda_mult,
                    )
                )
        return sorted(
            [
                item
                for sublist in await gather(*query_tasks)
                for item in sublist
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    @classmethod
    async def embed_file_to_vectorstore(
        cls,
        file: bytes,
        filename: str,
        collection_name: str,
    ) -> str:
        # if user uploads file, embed it
        try:
            text: str = await run_in_threadpool(
                read_bytes_to_text, file, filename
            )
            docs: list[str] = await VectorStoreManager.create_documents(
                text, collection_name=collection_name
            )
            doc_sample_without_triple_backticks: str = docs[0][:100].replace(
                "```", "'''"
            )
            return Lotties.OK.format(
                f'### Successfully Embedded\n`"{doc_sample_without_triple_backticks}"`'
            )
        except Exception:
            return Lotties.FAIL.format(
                "### Can't embed this type of file\nTry another file."
            )

    @staticmethod
    async def get_all_collection_names() -> list[str]:
        from qdrant_client import grpc

        grpc_collections = cache.vectorstore.client.async_grpc_collections
        response = await grpc_collections.List(grpc.ListCollectionsRequest())
        return [collection.name for collection in response.collections]

    @classmethod
    async def create_collection(
        cls,
        collection_name: str,
        distance: str = "cosine",
    ) -> None:
        """Create a collection, asynchronously.

        Distance can be one of:
        - cosine
        - euclid
        - dot"""

        from qdrant_client import grpc

        if distance.lower() == "cosine":
            distance = grpc.Distance.Cosine  # type: ignore
        elif distance.lower() == "euclid":
            distance = grpc.Distance.Euclid  # type: ignore
        elif distance.lower() == "dot":
            distance = grpc.Distance.Dot  # type: ignore
        else:
            raise ValueError(f"Unknown distance metric: {distance}")

        grpc_collections = cache.vectorstore.client.async_grpc_collections
        await grpc_collections.Create(
            grpc.CreateCollection(  # type: ignore
                collection_name=collection_name,
                vectors_config=grpc.VectorsConfig(
                    params=grpc.VectorParams(  # type: ignore
                        size=await cls.get_vector_size(),
                        distance=distance,
                    )
                ),  # type: ignore
            )
        )

    @staticmethod
    async def delete_collection(collection_name: str) -> bool:
        """Delete a collection, asynchronously."""

        from qdrant_client import grpc

        grpc_collections = cache.vectorstore.client.async_grpc_collections
        response = await grpc_collections.Delete(grpc.DeleteCollection(collection_name=collection_name))  # type: ignore
        return getattr(response, "result", False)

    @property
    def shared_collection_name(self) -> str:
        return cache.vectorstore.collection_name  # type: ignore


# if __name__ == "__main__":
#     import asyncio

#     from app.common.config import Config

#     sample_texts = (
#         "Neural network models can be quite powerful, effectively helping to identify pat"
#         "terns and uncover structure in a variety of different tasks, from language trans"
#         "lation to pathology to playing games. At the same time, neural models (as well a"
#         "s other kinds of machine learning models) can contain problematic biases in many"
#         " forms. For example, classifiers trained to detect rude, disrespectful, or unrea"
#         'sonable comments may be more likely to flag the sentence "I am gay" than "I am s'
#         'traight"; face classification models may not perform as well for women of color;'
#         " speech transcription may have higher error rates for African Americans than Whi"
#         "te Americans.\n\nMany pre-trained machine learning models are widely available for"
#         " developers to use -- for example, TensorFlow Hub recently launched its platform"
#         " publicly. It's important that when developers use these models in their applica"
#         "tions, they're aware of what biases they contain and how they might manifest in "
#         "those applications.\n\nHuman data encodes human biases by default. Being aware of "
#         "this is a good start, and the conversation around how to handle it is ongoing. A"
#         "t Google, we are actively researching unintended bias analysis and mitigation st"
#         "rategies because we are committed to making products that work well for everyone"
#         ". In this post, we'll examine a few text embedding models, suggest some tools fo"
#         "r evaluating certain forms of bias, and discuss how these issues matter when bui"
#         "lding applications.\n\nGiven a trained text embedding model, we can directly measu"
#         "re the associations the model has between words or phrases. Many of these associ"
#         "ations are expected and are helpful for natural language tasks. However, some as"
#         "sociations may be problematic or hurtful. For example, the ground-breaking paper"
#         ' by Bolukbasi et al. found that the vector-relationship between "man" and "woman'
#         '" was similar to the relationship between "physician" and "registered nurse" or '
#         '"shopkeeper" and "housewife" in the popular publicly-available word2vec embeddin'
#         "g trained on Google News text.\n\nThe Word Embedding Association Test (WEAT) was r"
#         "ecently proposed by Caliskan et al. [5] as a way to examine the associations in "
#         "word embeddings between concepts captured in the Implicit Association Test (IAT)"
#         ". We use the WEAT here as one way to explore some kinds of problematic associati"
#         "ons.\n\nThe WEAT test measures the degree to which a model associates sets of targ"
#         "et words (e.g., African American names, European American names, flowers, insect"
#         's) with sets of attribute words (e.g., "stable", "pleasant" or "unpleasant"). Th'
#         "e association between two given words is defined as the cosine similarity betwee"
#         "n the embedding vectors for the words.\n\nFor example, the target lists for the fi"
#         "rst WEAT test are types of flowers and insects, and the attributes are pleasant "
#         'words (e.g., "love", "peace") and unpleasant words (e.g., "hatred," "ugly"). The'
#         " overall test score is the degree to which flowers are more associated with the "
#         "pleasant words, relative to insects. A high positive score (the score can range "
#         "between 2.0 and -2.0) means that flowers are more associated with pleasant words"
#         ", and a high negative score means that insects are more associated with pleasant"
#         " words.\n\nWhile the first two WEAT tests proposed in Caliskan et al. measure asso"
#         "ciations that are of little social concern (except perhaps to entomologists), th"
#         "e remaining tests measure more problematic biases.\n"
#     )
#     sample_queries = [
#         "Why Google researched unintended bias analysis and mitigation strategies?",
#         "What is the WEAT test?",
#     ]

#     async def main():
#         # await vectorstore_manager.create_documents(texts=sample_texts, chunk_size=50)
#         collection_name = "abc"
#         results: list[list[Document]] | None = await gather(
#             *[
#                 VectorStoreManager.asimilarity_search(query=sample_query, collection_name=collection_name)
#                 for sample_query in sample_queries
#             ]
#         )
#         if results is None:
#             print("Index does not exist")
#             return
#         for idx, docs in enumerate(results):
#             print(f"Query: {sample_queries[idx]}\nDocs: {[doc.page_content for doc in docs]}\n\n")

#     cache.start(config=Config.get("test"))
#     asyncio.run(main())
