from typing import Optional

from app.common.lotties import Lotties
from app.utils.api.translate import Translator
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.websocket import SendToWebsocket


async def translate_callback(
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
        msg=Lotties.TRANSLATE.format("### Translating"),
        websocket=buffer.websocket,
        chat_room_id=buffer.current_chat_room_id,
        finish=False,
    )
    try:
        r = await Translator.translate(text=query, src_lang=src_lang, trg_lang=trg_lang)
        r_show = show_result_prefix + r if show_result_prefix is not None else r
        await SendToWebsocket.message(
            msg=Lotties.OK.format("### Finished translation")
            + (r_show if show_result else ""),
            websocket=buffer.websocket,
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
        return r
    except Exception:
        await SendToWebsocket.message(
            msg=Lotties.FAIL.format("### Failed translation"),
            websocket=buffer.websocket,
            chat_room_id=buffer.current_chat_room_id,
            finish=finish,
            wait_next_query=wait_next_query,
        )
        return None
