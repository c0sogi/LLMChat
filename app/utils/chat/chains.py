from typing import Any, Optional, Tuple

from fastapi.concurrency import run_in_threadpool
from langchain import PromptTemplate
from orjson import loads as orjson_loads

from app.common.config import config
from app.common.constants import (
    JSON_PATTERN,
    QueryTemplates,
    QueryBasedSearchTemplates,
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
    async def web_browsing_chain(
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
            query_to_search: Optional[str] = await cls._aget_query_to_search(
                query,
                query_template=QueryBasedSearchTemplates.QUERY__JSONIFY_WEB_BROWSING,
            )
            if query_to_search is None:
                query_to_search = query
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg="---\n{query_to_search}\n```\n".format(
                    query_to_search=query_to_search.replace("```", "'''")
                ),
                chat_room_id=buffer.current_chat_room_id,
                finish=False,
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
            msg=f"\n```lottie-search-web\n### Searching vectorstore\n",
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )
        try:
            query_to_search: Optional[str] = await cls._aget_query_to_search(
                query,
                query_template=QueryBasedSearchTemplates.QUERY__JSONIFY_VECTORSTORE,
            )
            if query_to_search is None:
                query_to_search = query
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg="---\n{query_to_search}\n```\n".format(
                    query_to_search=query_to_search.replace("```", "'''")
                ),
                chat_room_id=buffer.current_chat_room_id,
                finish=False,
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

    @staticmethod
    async def _aget_query_to_search(
        query: str, query_template: PromptTemplate
    ) -> Optional[str]:
        try:
            json_query = JSON_PATTERN.search(
                await Shared().openai_llm.apredict(query_template.format(query=query))  # type: ignore
            )
            if json_query is None:
                raise ValueError("Result is None")
            else:
                return orjson_loads(json_query.group())["query"]
        except Exception:
            return

    @staticmethod
    async def aget_json(
        query_template: PromptTemplate, **kwargs_to_format: str
    ) -> Optional[Any]:
        try:
            json_query = JSON_PATTERN.search(
                await Shared().openai_llm.apredict(
                    query_template.format(**kwargs_to_format)
                )
            )
            if json_query is None:
                raise ValueError("Result is None")
            else:
                return orjson_loads(json_query.group())
        except Exception:
            return
