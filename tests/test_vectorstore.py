import asyncio
from os import environ
from uuid import uuid4

import pytest
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import AgentAction, AgentFinish, HumanMessage, LLMResult
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import SequentialChain
from app.utils.langchain.llmchain import ConversationalRetrievalChain
from app.utils.langchain.redis_vectorstore import Redis, _aensure_index_exist, _redis_prefix
from app.utils.langchain.streaming_callbacks import StreamingCallbackHandler
from app.utils.langchain.vectorstore_memory import VectorStoreRetrieverMemory, ConversationTokenBufferMemory


@pytest.mark.asyncio
async def test_embedding(config, test_logger):
    sample_texts = [
        "Thornton Gombar lifted his gaze to the sky as a hovercraft zoomed over his home. Based on the rosy pink tints that adorned its exterior, Thornton surmised that the vehicle was a pleasure craft, likely filled with a small group of men and women engaging in gross debauchery as the self-driving vehicle propelled itself across the sky. The craft jerked and jolted in an arrhythmic manner as it flew to the heart of Homasoro City.",
        "The wind was a torrent of darkness among the gusty trees, The moon was a ghostly galleon tossed upon cloudy seas, The road was a ribbon of moonlight over the purple moor, And the highwayman came riding— Riding—riding— The highwayman came riding, up to the old inn-door.",
        "The sun shone brightly on the small village of Kinkan. It was a beautiful day, and everyone was busy with their daily chores. In the center of the village, there was a large fountain where children liked to play and splash each other. Among them was a young girl named Ahiru, who had bright red hair and a cheerful smile. She loved to dance and dreamed of becoming a ballerina someday.",
    ]  # noqa: E501
    sample_queries = [
        "What is the name of the city where Thornton lives?",
        "What is the rhyme scheme of “The Highwayman”?",
        "What is the genre of “Princess Tutu”?",
    ]
    text_splitter = TokenTextSplitter(
        chunk_size=20,
        chunk_overlap=0,
        model_name="gpt-3.5-turbo",
    )

    texts = text_splitter.split_text("\n".join(sample_texts))  # noqa: F841
    test_logger.info(f"len(texts): {len(texts)}")
    embeddings: Embeddings = OpenAIEmbeddings(openai_api_key=environ["OPENAI_API_KEY"])

    redis_url = "redis://{username}:{password}@{host}:{port}/{db}".format(
        username="",
        password=config.redis_password,
        host=config.redis_host,
        port=config.redis_port,
        db=config.redis_database,
    )
    # rds = await Redis.afrom_texts(
    #     texts,
    #     embedding=embeddings,
    #     redis_url=redis_url,
    #     index_name="test_texts",
    # )
    rds = await Redis.afrom_existing_index(embedding=embeddings, redis_url=redis_url, index_name="test_texts")
    test_logger.info(f"rds.index_name: {rds.index_name}")
    for query in sample_queries:
        result = await rds.asimilarity_search(query, k=1)
        test_logger.info(f"result: {result}")


@pytest.mark.asyncio
async def test_conversation_chain(config, test_logger):
    llm = ChatOpenAI(
        callback_manager=CallbackManager([StreamingCallbackHandler()]),
        streaming=True,
        temperature=0,
        max_tokens=500,
        verbose=True,
    )
    embeddings: Embeddings = OpenAIEmbeddings()
    memory = ConversationTokenBufferMemory(
        return_messages=True,
        llm=llm,
        memory_key="chat_history",
    )
    redis_url = "redis://{username}:{password}@{host}:{port}/{db}".format(
        username="",
        password=config.redis_password,
        host=config.redis_host,
        port=config.redis_port,
        db=config.redis_database,
    )
    redis = Redis(
        redis_url=redis_url,
        index_name="test",
        embedding_function=embeddings.embed_query,
        is_async=True,
    )
    redis_retriever = redis.as_retriever()

    await _aensure_index_exist(
        client=redis.client,
        index_name=redis.index_name,
        prefix=_redis_prefix(redis.index_name),
        dim=1536,
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=redis_retriever,
        memory=memory,
        verbose=True,
    )
    for query in ("What is capital city of South Korea?", "And what is the north's?"):
        response = await qa.acall({"question": query})
        test_logger.critical(f"Response: {response}")
    await redis.client.close()


def test_chat():
    from langchain.chains import ConversationChain
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    conversation.predict(input="Hi there!")
    # -> 'Hello! How can I assist you today?'

    conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
    # -> "That sounds like fun! I'm happy to chat with you. Is there anything specific you'd like to talk about?"

    conversation.predict(input="Tell me about yourself.")
    # -> "Sure! I am an AI language model created by OpenAI. I was trained on a large dataset of text from the internet, which allows me to understand and generate human-like language. I can answer questions, provide information, and even have conversations like this one. Is there anything else you'd like to know about me?"
