from typing import Any, Optional, Tuple
from aiohttp import ClientSession

from fastapi.concurrency import run_in_threadpool
from langchain import PromptTemplate
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
from app.utils.logger import ApiLogger


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
                json_query_template=JsonTemplates.QUERY__GET_QUERY_FOR_WEB_BROWSING,
            )
            r = await run_in_threadpool(Shared().duckduckgo.run, query_to_search)
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
                scrollable_contents: list[
                    str
                ] = Shared().token_text_splitter.split_text(
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
                    answerable_or_not_json = await Chains.aget_json(
                        query_template=JsonTemplates.CONTEXT_QUERY__ANSWERABLE_OR_NOT,
                        context=scrollable_content,
                        question=query,
                    )
                    if not isinstance(
                        answerable_or_not_json, dict
                    ) or answerable_or_not_json.get("answerable") not in (True, False):
                        await SendToWebsocket.message(
                            websocket=buffer.websocket,
                            msg=f"\n```lottie-fail\n### Reading content failed\n```\n",
                            chat_room_id=buffer.current_chat_room_id,
                            finish=False,
                        )
                        continue
                    if answerable_or_not_json["answerable"]:
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
        return_when_fail: Optional[str] = None
        try:
            query_to_search: str = await cls.aget_query_to_search(
                buffer=buffer,
                query=query,
                json_query_template=JsonTemplates.QUERY__GET_QUERY_FOR_WEB_BROWSING,
            )
            snippets_with_link: dict[str, str] = await run_in_threadpool(
                Shared().duckduckgo.formatted_results_with_link, query=query_to_search
            )
            return_when_fail = "\n\n".join(snippets_with_link.values())
            ApiLogger("|A06|").debug(snippets_with_link)

            while snippets_with_link:
                action_and_link_json = await Chains.aget_json(
                    query_template=JsonTemplates.CONTEXT_QUERY__CLICK_LINK_OR_FINISH,
                    query=query,
                    context="\n\n".join(snippets_with_link.values()),
                )
                if (
                    not isinstance(action_and_link_json, dict)
                    or (action_and_link_json.get("action") not in ("click", "finish"))
                    or (
                        action_and_link_json.get("action") == "link"
                        and action_and_link_json.get("link") not in snippets_with_link
                    )
                ):
                    return
                if action_and_link_json.get("action") == "finish":
                    await SendToWebsocket.message(
                        websocket=buffer.websocket,
                        msg=f"\n```lottie-ok\n### I found the result!\n```\n",
                        chat_room_id=buffer.current_chat_room_id,
                        finish=False,
                    )
                    return return_when_fail
                snippets_with_link.pop(action_and_link_json["link"])
                scroll_result: str | None = await scrolling(
                    link=action_and_link_json["link"],
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
                        msg=f"\n```lottie-fail\n### This link is not sufficient to answer the user's question.\n```\n",
                        chat_room_id=buffer.current_chat_room_id,
                        finish=False,
                    )

            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"\n```lottie-ok\n### Finished browsing, but I couldn't find the result.\n```\n",
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )
            return return_when_fail
        except Exception as e:
            ApiLogger("|A07|").error(e)
            await SendToWebsocket.message(
                msg=f"\n```lottie-fail\n### Failed to browse web\n```\n",
                websocket=buffer.websocket,
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )
            return return_when_fail

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
                json_query_template=JsonTemplates.QUERY__GET_QUERY_FOR_VECTORSTORE,
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
            context_and_query: str = (
                QueryTemplates.CONTEXT_QUESTION__CONTEXT_ONLY.format(
                    context=found_text, question=query
                )
            )
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"\n```lottie-ok\n### Finished searching vectorstore\n```\n{found_text if show_result else ''}",
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )
            return context_and_query

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
        json_query_template: PromptTemplate,
    ) -> str:
        query_to_search_json: Optional[dict] = await cls.aget_json(
            query_template=json_query_template,
            query=query,
        )
        if (
            not isinstance(query_to_search_json, dict)
            or "query" not in query_to_search_json
        ):
            query_to_search = query
        else:
            query_to_search = query_to_search_json["query"]
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
                await Shared().openai_llm.apredict(  # type: ignore
                    query_template.format(**kwargs_to_format)
                )
            )
            if json_query is None:
                raise ValueError("Result is None")
            else:
                return orjson_loads(json_query.group())
        except Exception:
            return
