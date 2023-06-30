from typing import Optional, Tuple

from app.common.config import ChatConfig, config
from app.common.lotties import Lotties
from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.vectorstore import Document, VectorStoreManager
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.logger import ApiLogger
from . import aget_query_to_search


async def vectorstore_query_chain(
    buffer: BufferedUserContext,
    query: str,
    finish: bool,
    wait_next_query: bool,
    show_result: bool = False,
    k: int = ChatConfig.vectorstore_n_results_limit,
) -> Optional[str]:
    await SendToWebsocket.message(
        websocket=buffer.websocket,
        msg=Lotties.SEARCH_DOC.format("### Searching vectorstore", end=False),
        chat_room_id=buffer.current_chat_room_id,
        finish=False,
    )
    try:
        query_to_search: str = await aget_query_to_search(
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
        # ApiLogger("||vectorstore_query_chain||").info(
        #     [score for _, score in found_text_and_score]
        # )
        if not found_text_and_score:
            raise Exception("No result found")

        found_text: Optional[str] = (
            "\n\n".join([document.page_content for document, _ in found_text_and_score])
            if found_text_and_score
            else None
        )
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=Lotties.OK.format("### Finished searching vectorstore")
            + (str(found_text) if show_result else ""),
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
        return found_text

    except Exception as e:
        ApiLogger("||vectorstore_query_chain||").exception(e)
        await SendToWebsocket.message(
            msg=Lotties.FAIL.format("### Failed to search vectorstore"),
            websocket=buffer.websocket,
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
