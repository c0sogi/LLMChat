from typing import Optional
from urllib.parse import urlparse

from requests_html import HTML, AsyncHTMLSession

from app.common.lotties import Lotties
from app.models.base_models import ParserDefinitions
from app.models.function_calling.functions import FunctionCalls
from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.logger import ApiLogger

from ..request import request_function_call

PARSER_DEFINITIONS = {
    "dcinside.com": ParserDefinitions(
        selector="//div[contains(@class, 'view_content_wrap') or contains(@class, 'comment_box')]",
        render_js=True,
    ),
}
BASE_FILTERING = (
    "[not(self::script or self::style or self::head or self::meta "
    "or self::link or self::title or self::noscript or self::iframe) and not(ancestor::script or ancestor::style)]"
)


async def _parse_text_content(
    url: str,
    asession: AsyncHTMLSession,
    tokens_per_chunk: int,
    chunk_overlap: int,
    timeout_for_render: int = 10,
    sleep_for_render: int = 0,
) -> list[str]:
    try:
        tld = ".".join(urlparse(url).netloc.split(".")[-2:])
        html: HTML = (await asession.get(url, timeout=10)).html  # type: ignore

        parser_definition = PARSER_DEFINITIONS.get(tld)
        js_required = parser_definition and parser_definition.render_js
        if js_required:
            await html.arender(
                timeout=timeout_for_render,
                sleep=sleep_for_render,
                keep_page=True,
            )
        if not parser_definition or not parser_definition.selector:
            if html.xpath("//article"):
                selector = "//article"
            else:
                selector = "//body"
        else:
            selector = parser_definition.selector

        contents = html.xpath(selector + "//text()" + BASE_FILTERING)
        if isinstance(contents, list):
            text = " ".join(
                [
                    content.strip()
                    for content in contents
                    if isinstance(content, str) and content.strip()
                ]
            )
        elif isinstance(contents, str):
            text = contents.strip()
        else:
            text = contents.text.strip()

        return Shared().token_text_splitter.split_text(
            text,
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
        )

    except Exception as e:
        ApiLogger("||_parse_text_content||").exception(e)
        return []


async def _get_controlling_page_and_relevance_score(
    query: str,
    link: str,
    scroll_position: int,
    max_scroll_position: int,
    previous_actions: list[str],
    scrollable_content: str,
    timeout: Optional[float] = 30,
) -> tuple[Optional[str], Optional[int]]:
    context = (
        f"Current link: {link}\nCurrent scroll bar: [{scroll_position}/{max_scroll_position}]\nYour "
        f"previous actions\n```{str(previous_actions)}```\nCurrent reading content\n```"
        f"{scrollable_content}```\n"
    )
    try:
        function = FunctionCalls.get_function_call(
            FunctionCalls.control_web_page
        )
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
        relevance_score = function_call_parsed["arguments"].get(
            "relevance_score"
        )

        if action in ("scroll_down", "go_back", "pick"):
            if (
                isinstance(relevance_score, str) and relevance_score.isdigit()
            ) or isinstance(relevance_score, int):
                return (str(action), int(relevance_score))
            else:
                return (str(action), None)
        else:
            raise ValueError("Invalid action")
    except Exception:
        return (None, None)


async def click_link_callback(
    buffer: BufferedUserContext,
    query: str,
    link: str,
    scrolling_chunk_size: int,
    scrolling_chunk_overlap: int,
    asession: Optional[AsyncHTMLSession] = None,
    timeout: float | int = 10,
    maximum_scrolls: int = 10,
) -> list[tuple[str, int]]:
    """Click link and get most relevant content and score"""
    if asession is None:
        asession = AsyncHTMLSession()
        temp_asession = True
    else:
        temp_asession = False
    try:
        content_idx_and_score_list: list[tuple[int, int]] = []
        previous_actions: list[str] = []
        # most_relevant_content_and_score: Optional[tuple[str, int]] = None
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=Lotties.CLICK.format(f"### Clicking link \n---\n{link}"),
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )
        # Get content from link
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=Lotties.READ.format("### Reading content"),
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )
        scrollable_contents: list[str] = await _parse_text_content(
            url=link,
            asession=asession,
            tokens_per_chunk=scrolling_chunk_size,
            chunk_overlap=scrolling_chunk_overlap,
        )
        max_scroll_position: int = len(scrollable_contents)
        for scroll_idx, scrollable_content in enumerate(
            scrollable_contents[:maximum_scrolls]
        ):
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
                content_idx_and_score_list.append(
                    (scroll_idx, relevance_score)
                )
            if action == "scroll_down":
                previous_actions.append("scroll_down")
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=Lotties.SCROLL_DOWN.format("### Scrolling down"),
                    chat_room_id=buffer.current_chat_room_id,
                    finish=False,
                )
                continue
            elif action == "go_back":
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=Lotties.GO_BACK.format("### Going back"),
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
            (scrollable_contents[idx], score)
            for idx, score in content_idx_and_score_list
        ]
    finally:
        if temp_asession:
            await asession.close()
