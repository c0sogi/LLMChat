from typing import Optional

from fastapi.concurrency import run_in_threadpool

from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.logger import ApiLogger

from . import aget_query_to_search


async def lite_web_browsing_chain(
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
        query_to_search: str = await aget_query_to_search(
            buffer=buffer,
            query=query,
            search_llm=Shared().web_search_llm,
        )
        r = await run_in_threadpool(Shared().duckduckgo.run, query_to_search)
        ApiLogger("||lite_web_browsing_chain||").info(r)

        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=f"\n```lottie-ok\n### Finished browsing\n```\n{r if show_result else ''}",
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
        return r
    except Exception as e:
        ApiLogger("||lite_web_browsing_chain||").exception(e)
        await SendToWebsocket.message(
            msg=f"\n```lottie-fail\n### Failed to browse web\n```\n",
            websocket=buffer.websocket,
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
