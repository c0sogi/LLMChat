import openai
from asyncio import gather
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import TokenTextSplitter
from app.errors.api_exceptions import Responses_500
from app.utils.langchain.redis_vectorstore import (
    Redis,
    Document,
    _ensure_index_exist,
    _redis_prefix,
)
from redis.asyncio import Redis as AsyncRedisType
from app.common.config import OPENAI_API_KEY, Config


class EmbeddingManager:
    def __init__(
        self,
        index_name: str,
        content_key: str = "content",
        metadata_key: str = "metadata",
        vector_key: str = "content_vector",
        vector_dimension: int = 1536,
        openai_api_key: str | None = OPENAI_API_KEY,
    ) -> None:
        self.embeddings: Embeddings = OpenAIEmbeddings(
            client=openai.Embedding,
            openai_api_key=openai_api_key,
        )

        self.index_name = index_name
        self.content_key = content_key
        self.metadata_key = metadata_key
        self.vector_key = vector_key
        self.vector_dimension = vector_dimension
        self.redis: Redis | None = None

    def start(self, config: Config) -> None:
        self.redis_url = config.redis_url_format.format(
            username="",
            password=config.redis_password,
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_database,
        )
        temp_rds = Redis(
            redis_url=self.redis_url,
            index_name=self.index_name,
            embedding_function=self.embeddings.embed_query,
            content_key=self.content_key,
            metadata_key=self.metadata_key,
            vector_key=self.vector_key,
            is_async=False,
        )
        _ensure_index_exist(
            client=temp_rds.client,  # type: ignore
            index_name=self.index_name,
            prefix=_redis_prefix(self.index_name),
            content_key=self.content_key,
            metadata_key=self.metadata_key,
            vector_key=self.vector_key,
            dim=self.vector_dimension,
        )
        temp_rds.client.close()
        self.redis = Redis(  # type: ignore
            redis_url=self.redis_url,
            index_name=self.index_name,
            embedding_function=self.embeddings.embed_query,
            content_key=self.content_key,
            metadata_key=self.metadata_key,
            vector_key=self.vector_key,
            is_async=True,
        )

    async def close(self) -> None:
        if self.redis is not None:
            assert isinstance(self.redis.client, AsyncRedisType)
            await self.redis.client.close()

    async def create_documents(
        self,
        texts: str,
        chunk_size: int,
        chunk_overlap: int = 0,
        tokenizer_model: str = "gpt-3.5-turbo",
    ) -> None:
        if self.redis is None:
            raise Responses_500.vectorestore_not_initialized
        await self.redis.aadd_texts(
            texts=TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_name=tokenizer_model,
            ).split_text(texts)
        )

    async def asimilarity_search(self, queries: list[str], k: int = 1) -> list[Document]:
        if self.redis is None:
            raise Responses_500.vectorestore_not_initialized
        return await gather(*[self.redis.asimilarity_search(query, k=k) for query in queries])


embedding_manager = EmbeddingManager(index_name="embeddings")

if __name__ == "__main__":
    import asyncio

    sample_texts = """Neural network models can be quite powerful, effectively helping to identify patterns and uncover structure in a variety of different tasks, from language translation to pathology to playing games. At the same time, neural models (as well as other kinds of machine learning models) can contain problematic biases in many forms. For example, classifiers trained to detect rude, disrespectful, or unreasonable comments may be more likely to flag the sentence "I am gay" than "I am straight"; face classification models may not perform as well for women of color; speech transcription may have higher error rates for African Americans than White Americans.

Many pre-trained machine learning models are widely available for developers to use -- for example, TensorFlow Hub recently launched its platform publicly. It's important that when developers use these models in their applications, they're aware of what biases they contain and how they might manifest in those applications.

Human data encodes human biases by default. Being aware of this is a good start, and the conversation around how to handle it is ongoing. At Google, we are actively researching unintended bias analysis and mitigation strategies because we are committed to making products that work well for everyone. In this post, we'll examine a few text embedding models, suggest some tools for evaluating certain forms of bias, and discuss how these issues matter when building applications.

Given a trained text embedding model, we can directly measure the associations the model has between words or phrases. Many of these associations are expected and are helpful for natural language tasks. However, some associations may be problematic or hurtful. For example, the ground-breaking paper by Bolukbasi et al. found that the vector-relationship between "man" and "woman" was similar to the relationship between "physician" and "registered nurse" or "shopkeeper" and "housewife" in the popular publicly-available word2vec embedding trained on Google News text.

The Word Embedding Association Test (WEAT) was recently proposed by Caliskan et al. [5] as a way to examine the associations in word embeddings between concepts captured in the Implicit Association Test (IAT). We use the WEAT here as one way to explore some kinds of problematic associations.

The WEAT test measures the degree to which a model associates sets of target words (e.g., African American names, European American names, flowers, insects) with sets of attribute words (e.g., "stable", "pleasant" or "unpleasant"). The association between two given words is defined as the cosine similarity between the embedding vectors for the words.

For example, the target lists for the first WEAT test are types of flowers and insects, and the attributes are pleasant words (e.g., "love", "peace") and unpleasant words (e.g., "hatred," "ugly"). The overall test score is the degree to which flowers are more associated with the pleasant words, relative to insects. A high positive score (the score can range between 2.0 and -2.0) means that flowers are more associated with pleasant words, and a high negative score means that insects are more associated with pleasant words.

While the first two WEAT tests proposed in Caliskan et al. measure associations that are of little social concern (except perhaps to entomologists), the remaining tests measure more problematic biases.

We used the WEAT score to examine several word embedding models: word2vec and GloVe (previously reported in Caliskan et al.), and three newly-released models available on the TensorFlow Hub platform -- nnlm-en-dim50, nnlm-en-dim128, and universal-sentence-encoder. """
    sample_queries = [
        "In the paper written by Bolukbasi, what fact was found?",
        "What is the WEAT test?",
    ]
    config = Config.get("test")

    async def main():
        await embedding_manager.create_documents(texts=sample_texts, chunk_size=50)
        results = await embedding_manager.asimilarity_search(queries=sample_queries)
        print(results)

    embedding_manager.start(config=config)
    asyncio.run(main())
