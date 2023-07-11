from typing import Optional

from fastapi.concurrency import run_in_threadpool

from app.common.lotties import Lotties
from app.models.function_calling.functions import FunctionCalls
from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.logger import ApiLogger

from ..query import aget_query_to_search


async def lite_web_browsing_callback(
    buffer: BufferedUserContext,
    query: str,
    finish: bool,
    wait_next_query: bool,
    show_result: bool = False,
) -> Optional[str]:
    await SendToWebsocket.message(
        websocket=buffer.websocket,
        msg=Lotties.SEARCH_WEB.format("### Browsing web", end=False),
        chat_room_id=buffer.current_chat_room_id,
        finish=False,
    )
    try:
        query_to_search: str = await aget_query_to_search(
            buffer=buffer,
            query=query,
            function=FunctionCalls.get_function_call(FunctionCalls.web_search),
        )
        r = await run_in_threadpool(Shared().duckduckgo.run, query_to_search)
        # ApiLogger("||lite_web_browsing_chain||").info(r)

        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=Lotties.OK.format("### Finished browsing")
            + (r if show_result else ""),
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
        return r
    except Exception as e:
        ApiLogger("||lite_web_browsing_chain||").exception(e)
        await SendToWebsocket.message(
            msg=Lotties.FAIL.format("### Failed to browse web"),
            websocket=buffer.websocket,
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
