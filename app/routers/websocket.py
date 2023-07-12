from asyncio import sleep

from fastapi import APIRouter, WebSocket

from app.common.config import API_ENV, HOST_MAIN, OPENAI_API_KEY
from app.database.crud import api_keys
from app.database.schemas.auth import Users
from app.errors.api_exceptions import Responses_400, Responses_401
from app.utils.chat.managers.stream import ChatStreamManager
from app.utils.chat.managers.websocket import SendToWebsocket
from app.viewmodels.status import ApiKeyStatus, UserStatus

router = APIRouter()


@router.websocket("/chat/{api_key}")
async def ws_chat(websocket: WebSocket, api_key: str):
    if OPENAI_API_KEY is None:
        raise Responses_400.not_supported_feature
    await websocket.accept()  # accept websocket
    if api_key != OPENAI_API_KEY and not API_ENV == "test":
        _api_key, _user = await api_keys.get_api_key_and_owner(
            access_key=api_key
        )
        if _user.status not in (UserStatus.active, UserStatus.admin):
            await SendToWebsocket.message(
                websocket=websocket,
                msg="Your account is not active",
                chat_room_id=" ",
            )
            await sleep(60)
            raise Responses_401.not_authorized
        if _api_key.status is not ApiKeyStatus.active:
            await SendToWebsocket.message(
                websocket=websocket,
                msg="Your api key is not active",
                chat_room_id=" ",
            )
            await sleep(60)
            raise Responses_401.not_authorized
    else:
        _user = Users(email=f"testaccount@{HOST_MAIN}")
    await ChatStreamManager.begin_chat(
        websocket=websocket,
        user=_user,
    )
