from asyncio import gather
from uuid import uuid4

import pytest
from langchain.docstore.document import Document
from app.database.connection import cache
from app.utils.chat.managers.vectorstore import VectorStoreManager


@pytest.mark.asyncio
async def test_embedding_single_index(config, test_logger):
    """Warning! This is expensive test!
    It costs a lot to embed the text"""

    cache.start(config=config)
    test_logger.info("Testing embedding")
    collection_name: str = uuid4().hex
    test_logger.info(f"Collection name: {collection_name}")
    sample_text = (
        "Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Member"
        "s of Congress and the Cabinet. Justices of the Supreme Court. My fellow American"
        "s.   Last year COVID-19 kept us apart. This year we are finally together again. "
        " Tonight, we meet as Democrats Republicans and Independents. But most importantl"
        "y as Americans.  With a duty to one another to the American people to the Consti"
        "tution.  And with an unwavering resolve that freedom will always triumph over ty"
        "ranny.  Six days ago, Russia’s Vladimir Putin sought to shake the foundations of"
        " the free world thinking he could make it bend to his menacing ways. But he badl"
        "y miscalculated. He thought he could roll into Ukraine and the world would roll "
        "over. Instead he met a wall of strength he never imagined.  He met the Ukrainian"
        " people.  From President Zelenskyy to every Ukrainian, their fearlessness, their"
        " courage, their determination, inspires the world.  Groups of citizens blocking "
        "tanks with their bodies. Everyone from students to retirees teachers turned sold"
        "iers defending their homeland.  In this struggle as President Zelenskyy said in "
        "his speech to the European Parliament “Light will win over darkness.” The Ukrain"
        "ian Ambassador to the United States is here tonight. Let each of us here tonight"
        " in this Chamber send an unmistakable signal to Ukraine and to the world.  Pleas"
        "e rise if you are able and show that, Yes, we the United States of America stand"
        " with the Ukrainian people. Throughout our history we’ve learned this lesson whe"
        "n dictators do not pay a price for their aggression they cause more chaos. They "
        "keep moving. And the costs and the threats to America and the world keep rising."
        "    That’s why the NATO Alliance was created to secure peace and stability in Eu"
        "rope after World War 2. The United States is a member along with 29 other nation"
        "s.  It matters. American diplomacy matters. American resolve matters.  Putin’s l"
        "atest attack on Ukraine was premeditated and unprovoked.  He rejected repeated e"
        "fforts at diplomacy.  He thought the West and NATO wouldn’t respond. And he thou"
        "ght he could divide us at home. Putin was wrong. We were ready.  Here is what we"
        " did.    We prepared extensively and carefully. We spent months building a coali"
        "tion of other freedom-loving nations from Europe and the Americas to Asia and Af"
        "rica to confront Putin. I spent countless hours unifying our European allies. We"
        " shared with the world in advance what we knew Putin was planning and precisely "
        "how he would try to falsely justify his aggression.   We countered Russia’s lies"
        " with truth.    And now that he has acted the free world is holding him accounta"
        "ble.  Along with twenty-seven members of the European Union including France, Ge"
        "rmany, Italy, as well as countries like the United Kingdom, Canada, Japan, Korea"
        ", Australia, New Zealand, and many others, even Switzerland. We are inflicting p"
        "ain on Russia and supporting the people of Ukraine. Putin is now isolated from t"
        "he world more than ever. Together with our allies –we are right now enforcing po"
        "werful economic sanctions. We are cutting off Russia’s largest banks from the in"
        "ternational financial system.   Preventing Russia’s central bank from defending "
        "the Russian Ruble making Putin’s $630 Billion “war fund” worthless."
    )
    sample_queries = [
        "What has been the response of the Ukrainian people to the Russian invasion, as depicted in the speech?",
        (
            "What preparations did the speaker mention were made to confront Putin's actions"
            ", and how does this reflect on the role of NATO and American diplomacy?"
        ),
        (
            "What are the specific economic sanctions mentioned in the speech that the United"
            " States and its allies are enforcing against Russia, and how do they aim to impa"
            "ct Russia's economy and Putin's 'war fund'?"
        ),
    ]
    empty: list[list[Document]] = await gather(
        *[
            VectorStoreManager.asimilarity_search(sample_query, collection_name=collection_name, k=3)
            for sample_query in sample_queries
        ]
    )
    assert all(len(result) == 0 for result in empty)

    await VectorStoreManager.create_documents(sample_text, collection_name=collection_name)
    results: list[list[Document]] | None = await gather(
        *[
            VectorStoreManager.asimilarity_search(sample_query, collection_name=collection_name, k=3)
            for sample_query in sample_queries
        ]
    )
    assert results is not None
    for i, result in enumerate(results):
        test_logger.info(f"\n### Query Result{i + 1}")
        for j, doc in enumerate(result):
            test_logger.info(f"-----> Document[{j + 1}]\n{doc.page_content}\n")


@pytest.mark.asyncio
async def test_embedding_multiple_index(config, test_logger):
    cache.start(config=config)
    test_logger.info("Testing embedding")
    collection_names: list[str] = [uuid4().hex for _ in range(2)]
    test_logger.info(f"Collection names: {collection_names}")
    texts_1 = ["Monkey loves banana", "Apple is red"]
    texts_2 = ["Banana is yellow", "Apple is green"]
    queries = ["Monkey loves banana", "Apple is red"]
    empty: list[list[Document]] = await gather(
        *[
            VectorStoreManager.asimilarity_search_multiple_collections(query, collection_names=collection_names, k=3)
            for query in queries
        ]
    )
    assert all(len(result) == 0 for result in empty)

    for collection_name, texts in zip(collection_names, [texts_1, texts_2]):
        for text in texts:
            await VectorStoreManager.create_documents(text, collection_name=collection_name)

    queries_results: list[list[tuple[Document, float]]] = await gather(
        *[
            VectorStoreManager.asimilarity_search_multiple_collections_with_score(
                query, collection_names=collection_names, k=3
            )
            for query in queries
        ]
    )
    for query, query_results in zip(queries, queries_results):
        for doc, score in query_results:
            test_logger.info(f"\n\n\n\nQuery={query}\nScore={score}\nContent={doc.page_content}")
    test_logger.info(f"\n\n\n\n\n\nTesting embedding: {queries_results}")


@pytest.mark.asyncio
async def test_embedding_multiple_index_2(config, test_logger):
    cache.start(config=config)
    test_logger.info("Testing embedding")
    collection_names: list[str] = [uuid4().hex for _ in range(2)]
    test_logger.info(f"Collection names: {collection_names}")
    texts_1 = ["Monkey loves banana", "Apple is red"]
    texts_2 = ["Banana is yellow", "Apple is green"]
    queries = ["Monkey loves banana", "Apple is red"]
    empty: list[list[Document]] = await gather(
        *[
            VectorStoreManager.asimilarity_search_multiple_collections(query, collection_names=collection_names, k=3)
            for query in queries
        ]
    )
    assert all(len(result) == 0 for result in empty)

    for collection_name, texts in zip(collection_names, [texts_1, texts_2]):
        for text in texts:
            await VectorStoreManager.create_documents(text, collection_name=collection_name)

    queries_results: list[list[tuple[Document, float]]] = await gather(
        *[
            VectorStoreManager.amax_marginal_relevance_search_multiple_collections_with_score(
                query, collection_names=collection_names, k=3
            )
            for query in queries
        ]
    )
    for query, query_results in zip(queries, queries_results):
        for doc, score in query_results:
            test_logger.info(f"\n\n\n\nQuery={query}\nScore={score}\nContent={doc.page_content}")
    test_logger.info(f"\n\n\n\n\n\nTesting embedding: {queries_results}")
