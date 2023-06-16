import asyncio
from typing import Optional

from fastapi.concurrency import run_in_threadpool
from langchain.schema import HumanMessage, SystemMessage
from lxml.etree import HTML
from requests_html import AsyncHTMLSession, HTMLResponse

from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.logger import ApiLogger


def _parse_text_content(
    html_response: HTMLResponse, tokens_per_chunk: int, chunk_overlap: int
) -> list[str]:
    try:
        text = " ".join(
            [
                content
                for content in HTML(html_response.text, parser=None).xpath(
                    "//*[not(self::script) and not(self::style)]/text()"
                )
                if content.strip()
            ]
        )
        return Shared().token_text_splitter.split_text(
            text,
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
        )
    except Exception:
        return []


async def _get_controlling_page_and_relevance_score(
    query: str,
    link: str,
    scroll_position: int,
    max_scroll_position: int,
    previous_actions: list[str],
    scrollable_content: str,
    timeout: float | int = 10,
) -> tuple[Optional[str], Optional[int]]:
    context = (
        f"Current link: {link}\nCurrent scroll bar: [{scroll_position}/{max_scroll_position}]\nYour "
        f"previous actions\n```{str(previous_actions)}```\nCurrent reading content\n```"
        f"{scrollable_content}```\n"
    )
    try:
        control_web_page_llm_result = await asyncio.wait_for(
            Shared().control_web_page_llm.agenerate(
                messages=[
                    [
                        SystemMessage(content=context),
                        HumanMessage(content=query),
                    ]
                ],
            ),
            timeout=timeout,
        )
        control_web_page_llm_output = control_web_page_llm_result.llm_output
        if control_web_page_llm_output is None:
            return (None, None)
        arguments = control_web_page_llm_output["function_calls"][0]["arguments"]
        action = arguments.get("action", None)
        relevance_score = arguments.get("relevance_score", None)

        if action in ("scroll_down", "go_back", "pick"):
            if (
                isinstance(relevance_score, str) and relevance_score.isdigit()
            ) or isinstance(relevance_score, int):
                return (action, int(relevance_score))
            else:
                return (action, None)
        else:
            return (None, None)
    except asyncio.TimeoutError:
        return (None, None)


async def click_link_chain(
    buffer: BufferedUserContext,
    query: str,
    link: str,
    scrolling_chunk_size: int,
    scrolling_chunk_overlap: int,
    asession: AsyncHTMLSession,
    timeout: float | int = 10,
    maximum_scrolls: int = 20,
) -> list[tuple[str, int]]:
    """Click link and get most relevant content and score"""
    content_idx_and_score_list: list[tuple[int, int]] = []
    previous_actions: list[str] = []
    # most_relevant_content_and_score: Optional[tuple[str, int]] = None
    await SendToWebsocket.message(
        websocket=buffer.websocket,
        msg=f"\n```lottie-click\n### Clicking link \n---\n {link}\n```\n",
        chat_room_id=buffer.current_chat_room_id,
        finish=False,
    )
    # Get content from link
    await SendToWebsocket.message(
        websocket=buffer.websocket,
        msg=f"\n```lottie-read\n### Reading content\n```\n",
        chat_room_id=buffer.current_chat_room_id,
        finish=False,
    )
    try:
        html_response: HTMLResponse = await asession.get(link, timeout=10)  # type: ignore
    except Exception as e:
        ApiLogger("||_click_link||").exception(e)
        return []
    scrollable_contents: list[str] = await run_in_threadpool(
        _parse_text_content,
        html_response=html_response,
        tokens_per_chunk=scrolling_chunk_size,
        chunk_overlap=scrolling_chunk_overlap,
    )
    max_scroll_position: int = len(scrollable_contents)
    if not 0 < max_scroll_position < maximum_scrolls:
        ApiLogger("||_click_link||").warning(
            f"No scrollable content for {link}: {max_scroll_position}"
        )
        return []
    for scroll_idx, scrollable_content in enumerate(scrollable_contents):
        # Read the content and predict next action, and evaluate relevance score of the content
        (
            action,
            relevance_score,
        ) = await _get_controlling_page_and_relevance_score(
            query=query,
            link=link,
            scroll_position=scroll_idx + 1,
            max_scroll_position=max_scroll_position,
            previous_actions=previous_actions,
            scrollable_content=scrollable_content.strip(),
            timeout=timeout,
        )
        ApiLogger("||_click_link||").info(
            (
                f"\n### Action: {action}\n\n"
                f"### Relevance Score: {relevance_score}\n\n"
                f"### scrollable_content: {scrollable_content}\n\n"
            )
        )
        if relevance_score is not None and (
            not content_idx_and_score_list
            or relevance_score >= content_idx_and_score_list[-1][1]
        ):
            content_idx_and_score_list.append((scroll_idx, relevance_score))
        if action == "scroll_down":
            previous_actions.append("scroll_down")
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"\n```lottie-scroll-down\n### Scrolling down\n```\n",
                chat_room_id=buffer.current_chat_room_id,
                finish=False,
            )
            continue
        elif action == "go_back":
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"\n```lottie-go-back\n### Going back\n```\n",
                chat_room_id=buffer.current_chat_room_id,
                finish=False,
            )
            return [
                (scrollable_contents[idx], score)
                for idx, score in content_idx_and_score_list
            ]
        elif action == "pick":
            content_idx_and_score_list.append((scroll_idx, 10))
            return [
                (scrollable_contents[idx], score)
                for idx, score in content_idx_and_score_list
            ]
        else:
            continue

    return [
        (scrollable_contents[idx], score) for idx, score in content_idx_and_score_list
    ]
