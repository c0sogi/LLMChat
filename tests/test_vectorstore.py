from uuid import uuid4

import pytest
from langchain.docstore.document import Document
from app.database.connection import cache
from app.utils.chat.vectorstore_manager import VectorStoreManager


@pytest.mark.asyncio
async def test_embedding(config, test_logger):
    cache.start(config=config)
    test_logger.info("Testing embedding")
    index_name: str = uuid4().hex
    test_logger.info(f"Index name: {index_name}")
    sample_text = """Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.   Last year COVID-19 kept us apart. This year we are finally together again.  Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans.  With a duty to one another to the American people to the Constitution.  And with an unwavering resolve that freedom will always triumph over tyranny.  Six days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. He thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined.  He met the Ukrainian people.  From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world.  Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.  In this struggle as President Zelenskyy said in his speech to the European Parliament “Light will win over darkness.” The Ukrainian Ambassador to the United States is here tonight. Let each of us here tonight in this Chamber send an unmistakable signal to Ukraine and to the world.  Please rise if you are able and show that, Yes, we the United States of America stand with the Ukrainian people. Throughout our history we’ve learned this lesson when dictators do not pay a price for their aggression they cause more chaos. They keep moving. And the costs and the threats to America and the world keep rising.    That’s why the NATO Alliance was created to secure peace and stability in Europe after World War 2. The United States is a member along with 29 other nations.  It matters. American diplomacy matters. American resolve matters.  Putin’s latest attack on Ukraine was premeditated and unprovoked.  He rejected repeated efforts at diplomacy.  He thought the West and NATO wouldn’t respond. And he thought he could divide us at home. Putin was wrong. We were ready.  Here is what we did.    We prepared extensively and carefully. We spent months building a coalition of other freedom-loving nations from Europe and the Americas to Asia and Africa to confront Putin. I spent countless hours unifying our European allies. We shared with the world in advance what we knew Putin was planning and precisely how he would try to falsely justify his aggression.   We countered Russia’s lies with truth.    And now that he has acted the free world is holding him accountable.  Along with twenty-seven members of the European Union including France, Germany, Italy, as well as countries like the United Kingdom, Canada, Japan, Korea, Australia, New Zealand, and many others, even Switzerland. We are inflicting pain on Russia and supporting the people of Ukraine. Putin is now isolated from the world more than ever. Together with our allies –we are right now enforcing powerful economic sanctions. We are cutting off Russia’s largest banks from the international financial system.   Preventing Russia’s central bank from defending the Russian Ruble making Putin’s $630 Billion “war fund” worthless."""  # noqa: E501
    sample_queries = [
        "What has been the response of the Ukrainian people to the Russian invasion, as depicted in the speech?",
        "What preparations did the speaker mention were made to confront Putin's actions, and how does this reflect on the role of NATO and American diplomacy?",
        "What are the specific economic sanctions mentioned in the speech that the United States and its allies are enforcing against Russia, and how do they aim to impact Russia's economy and Putin's 'war fund'?",
    ]
    assert await VectorStoreManager.asimilarity_search(sample_queries, index_name=index_name, k=3) is None
    await VectorStoreManager.create_documents(sample_text, index_name=index_name)
    results: list[list[Document]] | None = await VectorStoreManager.asimilarity_search(
        sample_queries, index_name=index_name, k=3
    )
    assert results is not None
    for i, result in enumerate(results):
        test_logger.info(f"\n### Query Result{i + 1}")
        for j, doc in enumerate(result):
            test_logger.info(f"-----> Document[{j + 1}]\n{doc.page_content}\n")
