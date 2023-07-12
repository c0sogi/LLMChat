from collections import deque
from copy import deepcopy
from re import Pattern, compile
from typing import Optional, Sequence, Tuple, cast

from fastapi.concurrency import run_in_threadpool
from requests_html import AsyncHTMLSession

from app.common.config import ChatConfig
from app.common.lotties import Lotties
from app.models.function_calling.base import FunctionCall, JsonTypes
from app.models.function_calling.functions import FunctionCalls
from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.logger import ApiLogger

from ..request import request_function_call
from .click_link import click_link_callback

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
    timeout: Optional[float] = 30,
) -> tuple[Optional[str], Optional[str]]:
    """Get action and link to click, based on snippets and most relevant content"""
    function: FunctionCall = deepcopy(
        FunctionCalls.get_function_call(FunctionCalls.control_browser)
    )
    snippets = "\n\n".join(snippets_with_link.values())

    # Define enum for link_to_click, which is one of the link of snippets or None
    assert function.parameters is not None, "function.parameters is None"
    for parameter in function.parameters:
        if "link" in parameter.name:
            parameter.enum = cast(
                list[JsonTypes], list(snippets_with_link.keys()) + ["null"]
            )

    # The context is the most relevant content and snippets
    context: str = (
        f"Most relevant content\n```{relevant_content_and_scores[-1][0]}```\nSnippets\n```{snippets}```"
        if relevant_content_and_scores
        else f"Snippets\n```{snippets}```"
    )
    # Get action and link to click from function call API
    function_call_parsed = await request_function_call(
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": query},
        ],
        functions=[function],
        function_call=function,
        timeout=timeout,
    )
    if "arguments" not in function_call_parsed:
        raise ValueError("No arguments returned")
    action = function_call_parsed["arguments"]["action"]
    link_to_click = function_call_parsed["arguments"].get(
        "link_to_click", "null"
    )
    if action == "click_link":
        if link_to_click == "null":
            return (None, None)
        return ("click_link", str(link_to_click))
    elif action == "finish_browsing":
        return ("finish_browsing", None)
    else:
        raise ValueError(f"Unknown action {action}")


async def full_web_browsing_callback(
    buffer: BufferedUserContext,
    query_to_search: str,
    user_provided_links: list[str],
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

    asession = AsyncHTMLSession()
    try:
        # Get query to search, and perform web search
        snippets_with_link: dict[str, str] = await run_in_threadpool(
            Shared().duckduckgo.formatted_results_with_link,
            query=query_to_search,
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
                query=query_to_search,
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
                click_results: list[
                    tuple[str, int]
                ] = await click_link_callback(
                    buffer=buffer,
                    query=query_to_search,
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
                            msg=Lotties.OK.format("### I found the result!"),
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
                        msg=Lotties.FAIL.format("### Reading content failed"),
                        chat_room_id=buffer.current_chat_room_id,
                        finish=False,
                    )
                    continue
            elif action == "finish_browsing":
                # A sufficient amount of information has been provided or there is no more information to find.
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=Lotties.OK.format("### Finishing browsing"),
                    chat_room_id=buffer.current_chat_room_id,
                    finish=False,
                )
                return get_best_result()

        # Exhausted all snippets. No more links to click
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=Lotties.OK.format(
                "### Finished browsing, but I couldn't find the result."
            ),
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
        return get_best_result()
    except Exception as e:
        ApiLogger("||full_web_browsing_chain||").exception(e)
        await SendToWebsocket.message(
            msg=Lotties.FAIL.format("### Failed to browse web"),
            websocket=buffer.websocket,
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
        return get_best_result()
    finally:
        await asession.close()
