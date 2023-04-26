from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.common.config import HOST_MAIN, OPENAI_API_KEY
from app.database.crud import api_keys
from app.database.schemas.auth import Users
from app.utils.logger import api_logger
from app.utils.chatgpt.chatgpt_stream_manager import begin_chat, SendToWebsocket

router = APIRouter()


@router.websocket("/chatgpt/{api_key}")
async def ws_chatgpt(websocket: WebSocket, api_key: str):
    try:
        await websocket.accept()  # accept websocket
        try:  # get user from api key
            if api_key != OPENAI_API_KEY:
                user: Users = await api_keys.get_api_key_owner(access_key=api_key)
            else:
                user: Users = Users(email=f"testaccount@{HOST_MAIN}")
        except Exception as exception:
            await SendToWebsocket.message(websocket=websocket, msg=f"유효하지 않은 API 키입니다. 연결을 종료합니다. ({exception})")
            return
        await begin_chat(
            websocket=websocket,
            user_id=user.email,
            openai_api_key=OPENAI_API_KEY,
        )
    except WebSocketDisconnect:
        ...
    except Exception as exception:
        api_logger.error(exception, exc_info=True)
        await SendToWebsocket.message(websocket=websocket, msg=f"알 수 없는 오류가 발생했습니다. 연결을 종료합니다. ({exception})")
