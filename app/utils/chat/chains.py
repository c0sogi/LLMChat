from copy import deepcopy
from typing import Any, Optional, Tuple

from aiohttp import ClientSession
from fastapi.concurrency import run_in_threadpool
from langchain import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from lxml import html, etree
from orjson import loads as orjson_loads

from app.common.config import ChatConfig, config
from app.common.constants import JSON_PATTERN
from app.models.chat_models import MessageHistory
from app.models.openai_functions import OpenAIFunctions
from app.shared import Shared
from app.utils.api.translate import Translator
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.prompts import (
    cutoff_message_histories,
    message_histories_to_list,
    openai_parse_method,
)
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
                search_llm=Shared().web_search_llm,
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
        scrolling_chunk_size: int = ChatConfig.scrolling_chunk_size_when_browsing,
        scrolling_chunk_overlap: int = ChatConfig.scrolling_overlap_when_browsing,
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
                scrollable_contents: list[str] = await run_in_threadpool(
                    cls.parse_text_content,
                    raw_html=await res.text(),
                    tokens_per_chunk=tokens_per_chunk,
                    chunk_overlap=chunk_overlap,
                )
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=f"\n```lottie-read\n### Reading content\n```\n",
                    chat_room_id=buffer.current_chat_room_id,
                    finish=False,
                )
                for scroll_idx, scrollable_content in enumerate(
                    scrollable_contents, start=1
                ):
                    scrollable_content = scrollable_content.strip()
                    control_web_page_llm_output = (
                        await Shared().control_web_page_llm.agenerate(
                            messages=[
                                [
                                    SystemMessage(content=f"Current link: {link}"),
                                    SystemMessage(
                                        content=f"Current scroll bar: [{scroll_idx}/{len(scrollable_contents)}]"
                                    ),
                                    SystemMessage(
                                        content=f"Previous snippets\n```{fallback_result}```"
                                    ),
                                    SystemMessage(
                                        content=f"Current reading content\n```{scrollable_content}```\n"
                                    ),
                                    HumanMessage(content=query),
                                ]
                            ],
                        )
                    ).llm_output
                    if control_web_page_llm_output is None:
                        await SendToWebsocket.message(
                            websocket=buffer.websocket,
                            msg=f"\n```lottie-fail\n### Reading content failed\n```\n",
                            chat_room_id=buffer.current_chat_room_id,
                            finish=False,
                        )
                        continue
                    action = control_web_page_llm_output["function_calls"][0][
                        "arguments"
                    ]["action"]
                    if action == "scroll_down":
                        await SendToWebsocket.message(
                            websocket=buffer.websocket,
                            msg=f"\n```lottie-scroll-down\n### Scrolling down\n```\n",
                            chat_room_id=buffer.current_chat_room_id,
                            finish=False,
                        )
                    elif action == "go_back":
                        await SendToWebsocket.message(
                            websocket=buffer.websocket,
                            msg=f"\n```lottie-go-back\n### Going back\n```\n",
                            chat_room_id=buffer.current_chat_room_id,
                            finish=False,
                        )
                        break
                    elif action == "finish_browsing":
                        return scrollable_content

                return None

        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=f"\n```lottie-search-web\n### Browsing web\n",
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )
        fallback_result: Optional[str] = None
        visited_links: set[str] = set()
        try:
            query_to_search = await cls.aget_query_to_search(
                buffer=buffer,
                query=query,
                search_llm=Shared().web_search_llm,
            )
            snippets_with_link: dict[str, str] = await run_in_threadpool(
                Shared().duckduckgo.formatted_results_with_link, query=query_to_search
            )
            fallback_result = "\n\n".join(snippets_with_link.values())
            ApiLogger("|A06|").debug(snippets_with_link)
            browsing_function = deepcopy(OpenAIFunctions.WEB_BROWSING)

            while snippets_with_link:
                snippets = "\n\n".join(snippets_with_link.values())
                browsing_function["parameters"]["properties"]["link_to_click"][
                    "enum"
                ] = list(snippets_with_link.keys()) + ["null"]
                browsing_llm_output = (
                    await Shared().browsing_llm.agenerate(
                        messages=[
                            [
                                SystemMessage(content=snippets),
                                HumanMessage(content=query),
                            ]
                        ],
                        functions=[browsing_function],
                        function_call={"name": browsing_function["name"]},
                    )
                ).llm_output
                if browsing_llm_output is None:
                    return fallback_result
                arguments = browsing_llm_output["function_calls"][0]["arguments"]

                if arguments["action"] == "click_link":
                    if arguments["link_to_click"] in visited_links:
                        continue
                        # raise Exception("Link already visited")
                    if arguments["link_to_click"] in snippets_with_link:
                        snippets_with_link.pop(arguments["link_to_click"])
                    elif arguments["link_to_click"] == "null":
                        continue
                    else:
                        visited_links.add(arguments["link_to_click"])
                    scroll_result: str | None = await scrolling(
                        link=arguments["link_to_click"],
                        tokens_per_chunk=scrolling_chunk_size,
                        chunk_overlap=scrolling_chunk_overlap,
                    )
                    if scroll_result is not None:
                        ApiLogger("|A08|").debug(scroll_result)
                        await SendToWebsocket.message(
                            websocket=buffer.websocket,
                            msg=f"\n```lottie-ok\n### I found the result!\n```\n",
                            chat_room_id=buffer.current_chat_room_id,
                            finish=False,
                        )
                        return scroll_result
                elif arguments["action"] == "finish_browsing":
                    await SendToWebsocket.message(
                        websocket=buffer.websocket,
                        msg=f"\n```lottie-ok\n### Finishing browsing\n```\n",
                        chat_room_id=buffer.current_chat_room_id,
                        finish=False,
                    )
                    return fallback_result

            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"\n```lottie-ok\n### Finished browsing, but I couldn't find the result.\n```\n",
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )
            return fallback_result
        except Exception as e:
            ApiLogger("|A07|").error(e)
            await SendToWebsocket.message(
                msg=f"\n```lottie-fail\n### Failed to browse web\n```\n",
                websocket=buffer.websocket,
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                wait_next_query=wait_next_query,
            )
            return fallback_result

    @classmethod
    async def vectorstore_query_chain(
        cls,
        buffer: BufferedUserContext,
        query: str,
        finish: bool,
        wait_next_query: bool,
        show_result: bool = False,
        k: int = ChatConfig.vectorstore_n_results_limit,
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
                search_llm=Shared().vectorstore_search_llm,
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
        user_message_histories, ai_message_histories = cutoff_message_histories(
            ai_message_histories=buffer.current_ai_message_histories,
            user_message_histories=buffer.current_user_message_histories
            + [
                MessageHistory(
                    role=buffer.current_user_chat_roles.user,
                    content=query,
                    tokens=buffer.current_user_chat_context.get_tokens_of(query),
                    is_user=True,
                )
            ],
            system_message_histories=[],
            token_limit=ChatConfig.query_context_token_limit,
        )
        web_search_llm_output = (
            await search_llm.agenerate(
                messages=[
                    message_histories_to_list(
                        user_chat_roles=buffer.current_user_chat_roles,
                        parse_method=openai_parse_method,
                        user_message_histories=user_message_histories,
                        ai_message_histories=ai_message_histories,
                    )
                ],
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
                await Shared().llm.apredict(  # type: ignore
                    query_template.format(**kwargs_to_format)
                )
            )
            if json_query is None:
                raise ValueError("Result is None")
            else:
                return orjson_loads(json_query.group())
        except Exception:
            return

    @staticmethod
    def parse_text_content(
        raw_html: str | bytes, tokens_per_chunk: int, chunk_overlap: int
    ) -> list[str]:
        try:
            paragraphs = html.fromstring(raw_html).xpath("//p")
        except etree.ParserError:
            return []
        return Shared().token_text_splitter.split_text(
            "\n".join([p.text_content().strip() for p in paragraphs]),
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
        )
