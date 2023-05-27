from asyncio import gather

from fastapi.concurrency import run_in_threadpool
from langchain.text_splitter import TokenTextSplitter
from redis.asyncio import Redis as AsyncRedisType
from app.common.config import EMBEDDING_VECTOR_DIMENSION

from app.database.connection import cache
from app.utils.chat.file_loader import read_bytes_to_text
from app.utils.langchain.redis_vectorstore import Document, _acheck_index_exists, _aensure_index_exist, _redis_prefix


async def search_if_index_exists(query: str, index_name: str, k: int) -> list[Document]:
    assert isinstance(cache.vectorstore.client, AsyncRedisType)
    if await _acheck_index_exists(client=cache.vectorstore.client, index_name=index_name):
        return await cache.vectorstore.asimilarity_search(query, index_name=index_name, k=k)
    else:
        return []


async def search_with_score_if_index_exists(query: str, index_name: str, k: int) -> list[tuple[Document, float]]:
    assert isinstance(cache.vectorstore.client, AsyncRedisType)
    if await _acheck_index_exists(client=cache.vectorstore.client, index_name=index_name):
        return await cache.vectorstore.asimilarity_search_with_score(query, index_name=index_name, k=k)
    else:
        return []


class VectorStoreManager:
    @staticmethod
    async def create_documents(
        text: str,
        index_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        dimension: int = EMBEDDING_VECTOR_DIMENSION,
        tokenizer_model: str = "gpt-3.5-turbo",
    ) -> list[str]:
        texts = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=tokenizer_model,
        ).split_text(text)
        assert isinstance(cache.vectorstore.client, AsyncRedisType)
        await _aensure_index_exist(
            client=cache.vectorstore.client,
            prefix=_redis_prefix(index_name),
            index_name=index_name,
            dim=dimension,
        )
        await cache.vectorstore.aadd_texts(texts, index_name=index_name)
        return texts

    @staticmethod
    async def asimilarity_search(
        queries: list[str],
        index_name: str,
        k: int = 1,
    ) -> list[list[Document]]:
        return await gather(*[search_if_index_exists(query=query, index_name=index_name, k=k) for query in queries])

    @staticmethod
    async def asimilarity_search_multiple_index(
        queries: list[str],
        index_names: list[str],
        k: int = 1,
    ) -> list[list[Document]]:
        # Nested function to handle index check and search.

        results = await gather(
            *[
                gather(*[search_if_index_exists(query=query, index_name=index_name, k=k) for query in queries])
                for index_name in index_names
            ]
        )  # shape: (index, query, k)

        # Reorganize results to have shape: (query, k * index)
        results = [[item for sublist in results_for_query for item in sublist] for results_for_query in zip(*results)]

        return results

    @staticmethod
    async def asimilarity_search_multiple_index_with_score(
        queries: list[str],
        index_names: list[str],
        k: int = 1,
    ) -> list[list[tuple[Document, float]]]:
        # Nested function to handle index check and search.

        # search_with_score_if_index_exists's shape: k
        # search_with_score_if_index_exists's return type: list[tuple[Document, float]]
        # Assume that tuple[Document, float] is a single result.
        results = await gather(
            *[
                gather(
                    *[search_with_score_if_index_exists(query=query, index_name=index_name, k=k) for query in queries]
                )
                for index_name in index_names
            ]
        )  # shape: (index, query, k)

        # Reorganize results to have shape: (query, k * index) and sort the results
        results = [
            sorted([item for sublist in results_for_query for item in sublist], key=lambda x: x[1])
            for results_for_query in zip(*results)
        ]

        return results

    @classmethod
    async def embed_file_to_vectorstore(
        cls,
        file: bytes,
        filename: str,
        index_name: str,
    ) -> str:
        # if user uploads file, embed it
        try:
            text: str = await run_in_threadpool(read_bytes_to_text, file, filename)
            docs: list[str] = await VectorStoreManager.create_documents(text, index_name=index_name)
            return f"Successfully embedded documents. You uploaded file begins with...\n\n```{docs[0][:50]}```..."
        except Exception:
            return "Can't embed this type of file. Try another file."

    @staticmethod
    async def drop_index(index_name: str) -> bool:
        assert isinstance(cache.vectorstore.client, AsyncRedisType)
        if not await _acheck_index_exists(client=cache.vectorstore.client, index_name=index_name):
            return False
        await cache.vectorstore.client.ft(index_name).dropindex(delete_documents=True)
        return True


if __name__ == "__main__":
    import asyncio

    from app.common.config import Config

    sample_texts = (
        "Neural network models can be quite powerful, effectively helping to identify pat"
        "terns and uncover structure in a variety of different tasks, from language trans"
        "lation to pathology to playing games. At the same time, neural models (as well a"
        "s other kinds of machine learning models) can contain problematic biases in many"
        " forms. For example, classifiers trained to detect rude, disrespectful, or unrea"
        'sonable comments may be more likely to flag the sentence "I am gay" than "I am s'
        'traight"; face classification models may not perform as well for women of color;'
        " speech transcription may have higher error rates for African Americans than Whi"
        "te Americans.\n\nMany pre-trained machine learning models are widely available for"
        " developers to use -- for example, TensorFlow Hub recently launched its platform"
        " publicly. It's important that when developers use these models in their applica"
        "tions, they're aware of what biases they contain and how they might manifest in "
        "those applications.\n\nHuman data encodes human biases by default. Being aware of "
        "this is a good start, and the conversation around how to handle it is ongoing. A"
        "t Google, we are actively researching unintended bias analysis and mitigation st"
        "rategies because we are committed to making products that work well for everyone"
        ". In this post, we'll examine a few text embedding models, suggest some tools fo"
        "r evaluating certain forms of bias, and discuss how these issues matter when bui"
        "lding applications.\n\nGiven a trained text embedding model, we can directly measu"
        "re the associations the model has between words or phrases. Many of these associ"
        "ations are expected and are helpful for natural language tasks. However, some as"
        "sociations may be problematic or hurtful. For example, the ground-breaking paper"
        ' by Bolukbasi et al. found that the vector-relationship between "man" and "woman'
        '" was similar to the relationship between "physician" and "registered nurse" or '
        '"shopkeeper" and "housewife" in the popular publicly-available word2vec embeddin'
        "g trained on Google News text.\n\nThe Word Embedding Association Test (WEAT) was r"
        "ecently proposed by Caliskan et al. [5] as a way to examine the associations in "
        "word embeddings between concepts captured in the Implicit Association Test (IAT)"
        ". We use the WEAT here as one way to explore some kinds of problematic associati"
        "ons.\n\nThe WEAT test measures the degree to which a model associates sets of targ"
        "et words (e.g., African American names, European American names, flowers, insect"
        's) with sets of attribute words (e.g., "stable", "pleasant" or "unpleasant"). Th'
        "e association between two given words is defined as the cosine similarity betwee"
        "n the embedding vectors for the words.\n\nFor example, the target lists for the fi"
        "rst WEAT test are types of flowers and insects, and the attributes are pleasant "
        'words (e.g., "love", "peace") and unpleasant words (e.g., "hatred," "ugly"). The'
        " overall test score is the degree to which flowers are more associated with the "
        "pleasant words, relative to insects. A high positive score (the score can range "
        "between 2.0 and -2.0) means that flowers are more associated with pleasant words"
        ", and a high negative score means that insects are more associated with pleasant"
        " words.\n\nWhile the first two WEAT tests proposed in Caliskan et al. measure asso"
        "ciations that are of little social concern (except perhaps to entomologists), th"
        "e remaining tests measure more problematic biases.\n"
    )
    sample_queries = [
        "Why Google researched unintended bias analysis and mitigation strategies?",
        "What is the WEAT test?",
    ]

    async def main():
        # await vectorstore_manager.create_documents(texts=sample_texts, chunk_size=50)
        index_name = "abc"
        results: list[list[Document]] | None = await VectorStoreManager.asimilarity_search(
            queries=sample_queries, index_name=index_name
        )
        if results is None:
            print("Index does not exist")
            return
        for idx, docs in enumerate(results):
            print(f"Query: {sample_queries[idx]}\nDocs: {[doc.page_content for doc in docs]}\n\n")

    cache.start(config=Config.get("test"))
    asyncio.run(main())
