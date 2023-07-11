from typing import Optional

from app.common.lotties import Lotties
from app.models.chat_models import ResponseType
from app.models.function_calling.functions import FunctionCalls
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.websocket import SendToWebsocket

from app.utils.function_calling.callbacks.translate import translate_callback
from app.utils.function_calling.query import aget_query_to_search


class BrowsingCommands:
    @staticmethod
    async def browse(
        user_query: str, /, buffer: BufferedUserContext, **kwargs
    ) -> tuple[Optional[str], ResponseType]:
        """Query LLM with duckduckgo browse results\n
        /browse <query>"""
        if user_query.startswith("/"):
            return user_query, ResponseType.REPEAT_COMMAND

        translate: Optional[str] = kwargs.get("translate", None)
        if translate:
            translate_chain_result: Optional[str] = await translate_callback(
                buffer=buffer,
                query=user_query,
                finish=False,
                wait_next_query=False,
                show_result=True,
                src_lang=translate,
                trg_lang="en",
            )
            if translate_chain_result is not None:
                user_query = translate_chain_result

        # Get query to search
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=Lotties.SEARCH_WEB.format("### Browsing web", end=False),
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )
        query_to_search = await aget_query_to_search(
            buffer=buffer,
            query=user_query,
            function=FunctionCalls.get_function_call(FunctionCalls.web_search),
        )

        return await FunctionCalls.web_search(
            query_to_search=query_to_search,
            buffer=buffer,
        )
