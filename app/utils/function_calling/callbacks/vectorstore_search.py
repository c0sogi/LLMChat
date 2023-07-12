from typing import Optional, Tuple

from app.common.config import ChatConfig, config
from app.common.lotties import Lotties
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.vectorstore import Document, VectorStoreManager
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.chat.tokens import make_truncated_text
from app.utils.logger import ApiLogger


async def vectorstore_search_callback(
    buffer: BufferedUserContext,
    query_to_search: str,
    finish: bool,
    wait_next_query: bool,
    show_result: bool = False,
    k: int = ChatConfig.vectorstore_n_results_limit,
) -> Optional[str]:
    try:
        found_text_and_score: list[
            Tuple[Document, float]
        ] = await VectorStoreManager.asimilarity_search_multiple_collections_with_score(
            query=query_to_search,
            collection_names=[buffer.user_id, config.shared_vectorestore_name],
            k=k,
        )
        if not found_text_and_score:
            raise Exception("No result found")

        found_text: Optional[str] = (
            "\n\n".join(
                [document.page_content for document, _ in found_text_and_score]
            )
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
        if found_text:
            found_text = make_truncated_text(
                user_chat_context=buffer.current_user_chat_context,
                text=found_text,
            )
        return found_text

    except Exception as e:
        ApiLogger("||vectorstore_search_callback||").exception(e)
        await SendToWebsocket.message(
            msg=Lotties.FAIL.format("### Failed to search vectorstore"),
            websocket=buffer.websocket,
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
        return None
