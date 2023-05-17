from asyncio import gather

from fastapi.concurrency import run_in_threadpool
from langchain.text_splitter import TokenTextSplitter

from app.database.connection import cache
from app.utils.chat.file_loader import read_bytes_to_text
from app.utils.langchain.redis_vectorstore import Document


class VectorStoreManager:
    @staticmethod
    async def create_documents(
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        tokenizer_model: str = "gpt-3.5-turbo",
    ) -> list[str]:
        texts = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=tokenizer_model,
        ).split_text(text)
        await cache.vectorstore.aadd_texts(texts=texts)
        return texts

    @staticmethod
    async def asimilarity_search(queries: list[str], k: int = 1) -> list[list[Document]]:
        return await gather(*[cache.vectorstore.asimilarity_search(query, k=k) for query in queries])

    @classmethod
    async def embed_file_to_vectorstore(cls, file: bytes, filename: str) -> str:
        # if user uploads file, embed it
        try:
            text: str = await run_in_threadpool(read_bytes_to_text, file, filename)
            docs: list[str] = await VectorStoreManager.create_documents(text)
            return f"Successfully embedded documents. You uploaded file begins with...\n\n```{docs[0][:50]}```..."
        except Exception:
            return "Can't embed this type of file. Try another file."


if __name__ == "__main__":
    import asyncio

    from app.common.config import Config

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
        "Why Google researched unintended bias analysis and mitigation strategies?",
        "What is the WEAT test?",
    ]

    async def main():
        # await vectorstore_manager.create_documents(texts=sample_texts, chunk_size=50)
        results: list[list[Document]] = await VectorStoreManager.asimilarity_search(queries=sample_queries)
        for idx, docs in enumerate(results):
            print(f"Query: {sample_queries[idx]}\nDocs: {[doc.page_content for doc in docs]}\n\n")

    cache.start(config=Config.get("test"))
    asyncio.run(main())
