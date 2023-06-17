from collections import deque
from copy import deepcopy
from re import Pattern, compile
from typing import Optional, Sequence, Tuple

from fastapi.concurrency import run_in_threadpool
from langchain.schema import HumanMessage, SystemMessage
from requests_html import AsyncHTMLSession

from app.common.config import ChatConfig
from app.models.openai_functions import OpenAIFunctions
from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.chains.click_link import click_link_chain
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.logger import ApiLogger

from . import aget_query_to_search

URL_PATTERN: Pattern = compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


def _harvest_click_results(
    click_results: list[tuple[str, int]],
    relevant_content_and_scores: deque[tuple[str, int]],
) -> None:
    """Harvest click results from a list of click results, and append to relevant_content_and_scores"""
    for scroll_content, score in click_results:
        if (
            not relevant_content_and_scores
            or score >= relevant_content_and_scores[-1][1]
        ):
            relevant_content_and_scores.append((scroll_content, score))


async def _get_browsing_action_and_link_to_click(
    query: str,
    snippets_with_link: dict[str, str],
    relevant_content_and_scores: Sequence[tuple[str, int]],
) -> tuple[Optional[str], Optional[str]]:
    """Get action and link to click, based on snippets and most relevant content"""
    browsing_function = deepcopy(OpenAIFunctions.WEB_BROWSING)
    snippets = "\n\n".join(snippets_with_link.values())
    # Define enum for link_to_click, which is one of the link of snippets or None
    browsing_function["parameters"]["properties"]["link_to_click"]["enum"] = list(
        snippets_with_link.keys()
    ) + ["null"]
    # The context is the most relevant content and snippets
    context: str = (
        f"Most relevant content\n```{relevant_content_and_scores[-1][0]}```\nSnippets\n```{snippets}```"
        if relevant_content_and_scores
        else f"Snippets\n```{snippets}```"
    )
    # Get action and link to click from function call API
    browsing_llm_output = (
        await Shared().browsing_llm.agenerate(
            messages=[
                [
                    SystemMessage(content=context),
                    HumanMessage(content=query),
                ]
            ],
            functions=[browsing_function],
            function_call={"name": browsing_function["name"]},
        )
    ).llm_output
    if browsing_llm_output is None:
        return (None, None)
    arguments = browsing_llm_output["function_calls"][0]["arguments"]
    action = arguments.get("action")
    link_to_click = arguments.get("link_to_click", "null")
    if action == "click_link":
        if link_to_click == "null":
            return (None, None)
        return ("click_link", link_to_click)
    elif action == "finish_browsing":
        return ("finish_browsing", None)
    else:
        return (None, None)


async def full_web_browsing_chain(
    buffer: BufferedUserContext,
    query: str,
    finish: bool,
    wait_next_query: bool,
    show_result: bool = True,
    scrolling_chunk_size: int = ChatConfig.scrolling_chunk_size_when_browsing,
    scrolling_chunk_overlap: int = ChatConfig.scrolling_overlap_when_browsing,
    num_content_chunks: int = 3,
    collect_score_larger_than: int = 5,
    timeout: float | int = 10,
    maximum_scrolls: int = 20,
) -> Optional[str]:
    # Init variables
    relevant_content_and_scores: deque[Tuple[str, int]] = deque(
        maxlen=num_content_chunks
    )
    visited_links: set[str] = set()
    user_provided_links: list[str] = URL_PATTERN.findall(query)
    overall_snippets: Optional[str] = None

    def get_best_result():
        # Get best result from relevant_content_and_scores and overall_snippets
        if not relevant_content_and_scores:
            # If no relevant content, return overall_snippets
            # ApiLogger("||get_best_result||").info("[1]" + str(overall_snippets))
            return overall_snippets
        elif relevant_content_and_scores[-1][1] > 5:
            # If there's content with more than `collect_content_larger_than`, return it
            best_results: list[str] = [
                i[0]
                for i in relevant_content_and_scores
                if i[1] > collect_score_larger_than
            ]
            if overall_snippets is not None:
                if not user_provided_links:
                    if len(best_results) < num_content_chunks:
                        best_results.insert(0, overall_snippets)
                    else:
                        best_results[0] = overall_snippets
            # ApiLogger("||get_best_result||").info("[2]" + "\n\n".join(best_results))
            return "\n\n".join(best_results)
        else:
            # Otherwise, return overall_snippets
            # ApiLogger("||get_best_result||").info("[3]" + str(overall_snippets))
            return overall_snippets

    # Begin browsing
    await SendToWebsocket.message(
        websocket=buffer.websocket,
        msg=f"\n```lottie-search-web\n### Browsing web\n",
        chat_room_id=buffer.current_chat_room_id,
        finish=False,
    )
    asession = AsyncHTMLSession()
    try:
        # Get query to search, and perform web search
        query_to_search = await aget_query_to_search(
            buffer=buffer,
            query=query,
            search_llm=Shared().web_search_llm,
        )
        snippets_with_link: dict[str, str] = await run_in_threadpool(
            Shared().duckduckgo.formatted_results_with_link, query=query_to_search
        )
        for link in user_provided_links:
            snippets_with_link[link] = (
                "User provided link"
                if link not in snippets_with_link
                else snippets_with_link[link]
            )
        overall_snippets = "\n\n".join(snippets_with_link.values())
        while snippets_with_link:
            # Get next action and link to click until no more snippets
            # Get action and link to click, based on snippets and most relevant content
            (
                action,
                link_to_click,
            ) = await _get_browsing_action_and_link_to_click(
                query=query,
                snippets_with_link=snippets_with_link,
                relevant_content_and_scores=relevant_content_and_scores,
            )
            ApiLogger("||full_web_browsing_chain||").info(
                f"\n>>> Action: {action}, Link to Click: {link_to_click}"
            )

            if action == "click_link":
                # Prepare to click link, prevent infinite loop
                if link_to_click is None or link_to_click in visited_links:
                    continue
                if link_to_click in snippets_with_link:
                    snippets_with_link.pop(link_to_click)
                else:
                    visited_links.add(link_to_click)

                # Get click result and most relevant content & score
                # Score: 0 = not relevant, 10 = most relevant
                click_results: list[tuple[str, int]] = await click_link_chain(
                    buffer=buffer,
                    query=query,
                    link=link_to_click,
                    scrolling_chunk_size=scrolling_chunk_size,
                    scrolling_chunk_overlap=scrolling_chunk_overlap,
                    asession=asession,
                    timeout=timeout,
                    maximum_scrolls=maximum_scrolls,
                )
                if click_results:
                    _harvest_click_results(
                        click_results=click_results,
                        relevant_content_and_scores=relevant_content_and_scores,
                    )
                    if click_results[-1][1] == 10:
                        # Found the best result!
                        await SendToWebsocket.message(
                            websocket=buffer.websocket,
                            msg=f"\n```lottie-ok\n### I found the result!\n```\n",
                            chat_room_id=buffer.current_chat_room_id,
                            finish=False,
                        )
                        return get_best_result()
                    else:
                        # Update most relevant content and continue browsing
                        continue
                else:
                    # Click failed, continue browsing
                    await SendToWebsocket.message(
                        websocket=buffer.websocket,
                        msg=f"\n```lottie-fail\n### Reading content failed\n```\n",
                        chat_room_id=buffer.current_chat_room_id,
                        finish=False,
                    )
                    continue
            elif action == "finish_browsing":
                # A sufficient amount of information has been provided or there is no more information to find.
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=f"\n```lottie-ok\n### Finishing browsing\n```\n",
                    chat_room_id=buffer.current_chat_room_id,
                    finish=False,
                )
                return get_best_result()

        # Exhausted all snippets. No more links to click
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=f"\n```lottie-ok\n### Finished browsing, but I couldn't find the result.\n```\n",
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
        return get_best_result()
    except Exception as e:
        ApiLogger("||full_web_browsing_chain||").exception(e)
        await SendToWebsocket.message(
            msg=f"\n```lottie-fail\n### Failed to browse web\n```\n",
            websocket=buffer.websocket,
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
        return get_best_result()
    finally:
        await asession.close()
