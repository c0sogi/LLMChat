from typing import Any, Optional, Tuple
from aiohttp import ClientSession

from fastapi.concurrency import run_in_threadpool
from langchain import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from lxml import html
from orjson import loads as orjson_loads

from app.common.config import config
from app.common.constants import (
    JSON_PATTERN,
    QueryTemplates,
    JsonTemplates,
)
from app.shared import Shared
from app.utils.api.translate import Translator
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.vectorstore_manager import Document, VectorStoreManager
from app.utils.chat.websocket_manager import SendToWebsocket
from app.utils.langchain.chat_openai import CustomChatOpenAI
from app.utils.logger import ApiLogger

shared = Shared()


class Chains:
    @staticmethod
    async def translate_chain(
        buffer: BufferedUserContext,
        query: str,
        finish: bool,
        wait_next_query: Optional[bool],
        show_result: bool = True,
        show_result_prefix: Optional[str] = " # 🌐 Translation Result\n---\n\n",
        src_lang: str = "en",
        trg_lang: str = "en",
    ) -> Optional[str]:
        await SendToWebsocket.message(
            msg=f"\n```lottie-translate\n### Translating\n```\n",
            websocket=buffer.websocket,
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )
        try:
            r = await Translator.translate(
                text=query, src_lang=src_lang, trg_lang=trg_lang
            )
            r_show = show_result_prefix + r if show_result_prefix is not None else r
            await SendToWebsocket.message(
                msg=f"\n```lottie-ok\n### Finished translation\n```\n{r_show if show_result else ''}",
                websocket=buffer.websocket,
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )
            return r
        except Exception:
            await SendToWebsocket.message(
                msg=f"\n```lottie-fail\n### Failed translation\n```\n",
                websocket=buffer.websocket,
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )

    @classmethod
    async def lite_web_browsing_chain(
        cls,
        buffer: BufferedUserContext,
        query: str,
        finish: bool,
        wait_next_query: bool,
        show_result: bool = False,
    ) -> Optional[str]:
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=f"\n```lottie-search-web\n### Browsing web\n",
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )
        try:
            query_to_search: str = await cls.aget_query_to_search(
                buffer=buffer,
                query=query,
                search_llm=shared.web_search_llm,
            )
            r = await run_in_threadpool(shared.duckduckgo.run, query_to_search)
            ApiLogger("|A03|").debug(r)

            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"\n```lottie-ok\n### Finished browsing\n```\n{r if show_result else ''}",
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )
            return r
        except Exception as e:
            ApiLogger("|A04|").error(e)
            await SendToWebsocket.message(
                msg=f"\n```lottie-fail\n### Failed to browse web\n```\n",
                websocket=buffer.websocket,
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )

    @classmethod
    async def full_web_browsing_chain(
        cls,
        buffer: BufferedUserContext,
        query: str,
        finish: bool,
        wait_next_query: bool,
        show_result: bool = True,
        tokens_per_chunk: int = 1024,
        chunk_overlap: int = 256,
    ) -> Optional[str]:
        async def scrolling(
            link: str,
            tokens_per_chunk: int,
            chunk_overlap: int,
        ) -> str | None:
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"\n```lottie-click\n### Clicking link \n---\n {link}\n```\n",
                chat_room_id=buffer.current_chat_room_id,
                finish=False,
            )
            async with ClientSession() as session:
                res = await session.get(link)
                paragraphs = html.fromstring(await res.read()).xpath("//p")
                scrollable_contents: list[str] = shared.token_text_splitter.split_text(
                    "\n".join([p.text_content().strip() for p in paragraphs]),
                    tokens_per_chunk=tokens_per_chunk,
                    chunk_overlap=chunk_overlap,
                )
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=f"\n```lottie-read\n### Reading content\n```\n",
                    chat_room_id=buffer.current_chat_room_id,
                    finish=False,
                )
                for scrollable_content in scrollable_contents:
                    scrollable_content = scrollable_content.strip()
                    answerable_or_not_llm_output = (
                        await shared.answerable_or_not_llm.agenerate(
                            messages=[
                                [
                                    SystemMessage(content=scrollable_content),
                                    HumanMessage(content=query),
                                ]
                            ],
                        )
                    ).llm_output
                    if answerable_or_not_llm_output is None:
                        await SendToWebsocket.message(
                            websocket=buffer.websocket,
                            msg=f"\n```lottie-fail\n### Reading content failed\n```\n",
                            chat_room_id=buffer.current_chat_room_id,
                            finish=False,
                        )
                        continue
                    answerable = answerable_or_not_llm_output["function_calls"][0][
                        "arguments"
                    ]["answerable"]
                    if answerable:
                        return scrollable_content
                    await SendToWebsocket.message(
                        websocket=buffer.websocket,
                        msg=f"\n```lottie-scroll-down\n### Scrolling down\n```\n",
                        chat_room_id=buffer.current_chat_room_id,
                        finish=False,
                    )
                return None

        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=f"\n```lottie-search-web\n### Browsing web\n",
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )
        snippets: Optional[str] = None
        visited_links: set[str] = set()
        try:
            web_search_llm_output = (
                await shared.web_search_llm.agenerate(
                    messages=[[HumanMessage(content=query)]],
                )
            ).llm_output
            if web_search_llm_output is None:
                return snippets
            query_to_search = await cls.aget_query_to_search(
                buffer=buffer,
                query=query,
                search_llm=shared.web_search_llm,
            )
            snippets_with_link: dict[str, str] = await run_in_threadpool(
                shared.duckduckgo.formatted_results_with_link, query=query_to_search
            )
            snippets = "\n\n".join(snippets_with_link.values())
            ApiLogger("|A06|").debug(snippets_with_link)

            while snippets_with_link:
                browsing_llm_output = (
                    await shared.browsing_llm.agenerate(
                        messages=[
                            [
                                SystemMessage(content=snippets),
                                HumanMessage(content=query),
                            ]
                        ],
                    )
                ).llm_output
                if browsing_llm_output is None:
                    return snippets
                arguments = browsing_llm_output["function_calls"][0]["arguments"]

                if arguments["action"] == "click_link":
                    if arguments["link_to_click"] in visited_links:
                        raise Exception("Link already visited")
                    if arguments["link_to_click"] in snippets_with_link:
                        snippets_with_link.pop(arguments["link_to_click"])
                    else:
                        visited_links.add(arguments["link_to_click"])
                    scroll_result: str | None = await scrolling(
                        link=arguments["link_to_click"],
                        tokens_per_chunk=tokens_per_chunk,
                        chunk_overlap=chunk_overlap,
                    )
                    if scroll_result is not None:
                        await SendToWebsocket.message(
                            websocket=buffer.websocket,
                            msg=f"\n```lottie-ok\n### I found the result!\n```\n",
                            chat_room_id=buffer.current_chat_room_id,
                            finish=False,
                        )
                        return scroll_result
                    else:
                        await SendToWebsocket.message(
                            websocket=buffer.websocket,
                            msg=f"\n```lottie-fail\n### This link is not sufficient to answer\n```\n",
                            chat_room_id=buffer.current_chat_room_id,
                            finish=False,
                        )
                elif arguments["action"] == "finish_browsing":
                    await SendToWebsocket.message(
                        websocket=buffer.websocket,
                        msg=f"\n```lottie-ok\n### I found the result!\n```\n",
                        chat_room_id=buffer.current_chat_room_id,
                        finish=False,
                    )
                    return snippets
                else:
                    raise Exception("Unknown action")

            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"\n```lottie-ok\n### Finished browsing, but I couldn't find the result.\n```\n",
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )
            return snippets
        except Exception as e:
            ApiLogger("|A07|").error(e)
            await SendToWebsocket.message(
                msg=f"\n```lottie-fail\n### Failed to browse web\n```\n",
                websocket=buffer.websocket,
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )
            return snippets

    @classmethod
    async def vectorstore_query_chain(
        cls,
        buffer: BufferedUserContext,
        query: str,
        finish: bool,
        wait_next_query: bool,
        show_result: bool = False,
        k: int = 3,
    ) -> Optional[str]:
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=f"\n```lottie-search-doc\n### Searching vectorstore\n",
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )
        try:
            query_to_search: str = await cls.aget_query_to_search(
                buffer=buffer,
                query=query,
                search_llm=shared.vectorstore_search_llm,
            )
            found_text_and_score: list[
                Tuple[Document, float]
            ] = await VectorStoreManager.asimilarity_search_multiple_collections_with_score(
                query=query_to_search,
                collection_names=[buffer.user_id, config.shared_vectorestore_name],
                k=k,
            )  # lower score is the better!
            ApiLogger("|A02|").debug([score for _, score in found_text_and_score])
            if not found_text_and_score:
                raise Exception("No result found")

            found_text: str = "\n\n".join(
                [document.page_content for document, _ in found_text_and_score]
            )
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"\n```lottie-ok\n### Finished searching vectorstore\n```\n{found_text if show_result else ''}",
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )
            return found_text

        except Exception as e:
            ApiLogger("|A05|").exception(e)
            await SendToWebsocket.message(
                msg=f"\n```lottie-fail\n### Failed to search vectorstore\n```\n",
                websocket=buffer.websocket,
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )

    @classmethod
    async def aget_query_to_search(
        cls,
        buffer: BufferedUserContext,
        query: str,
        search_llm: CustomChatOpenAI,
    ) -> str:
        web_search_llm_output = (
            await search_llm.agenerate(
                messages=[[HumanMessage(content=query)]],
            )
        ).llm_output
        if web_search_llm_output is None:
            return query
        query_to_search = web_search_llm_output["function_calls"][0]["arguments"][
            "query_to_search"
        ]
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg="---\n{query_to_search}\n```\n".format(
                query_to_search=query_to_search.replace("```", "'''")
            ),
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )
        return query_to_search

    @staticmethod
    async def aget_json(
        query_template: PromptTemplate, **kwargs_to_format: str
    ) -> Optional[Any]:
        try:
            json_query = JSON_PATTERN.search(
                await shared.llm.apredict(  # type: ignore
                    query_template.format(**kwargs_to_format)
                )
            )
            if json_query is None:
                raise ValueError("Result is None")
            else:
                return orjson_loads(json_query.group())
        except Exception:
            return
