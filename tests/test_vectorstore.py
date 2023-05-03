import openai
from os import environ

import pytest
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import TokenTextSplitter
from app.utils.langchain.redis_vectorstore import Redis, _aensure_index_exist, _redis_prefix


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
    embeddings: Embeddings = OpenAIEmbeddings(
        client=openai.Embedding,
        openai_api_key=environ["OPENAI_API_KEY"],
    )

    redis_url = "redis://{username}:{password}@{host}:{port}/{db}".format(
        username="",
        password=config.redis_password,
        host=config.redis_host,
        port=config.redis_port,
        db=config.redis_database,
    )
    rds = await Redis.afrom_texts(
        texts,
        embedding=embeddings,
        redis_url=redis_url,
        index_name="test_texts",
    )
    # rds = await Redis.afrom_existing_index(embedding=embeddings, redis_url=redis_url, index_name="test_texts")
    test_logger.info(f"rds.index_name: {rds.index_name}")
    for query in sample_queries:
        result = await rds.asimilarity_search(query, k=1)
        test_logger.info(f"result: {result}")
