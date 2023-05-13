from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.common.config import HOST_MAIN, OPENAI_API_KEY
from app.database.crud import api_keys
from app.database.schemas.auth import Users
from app.errors.api_exceptions import Responses_400
from app.utils.logger import api_logger
from app.utils.chatgpt.chatgpt_stream_manager import begin_chat
from app.common.config import API_ENV

router = APIRouter()


@router.websocket("/chatgpt/{api_key}")
async def ws_chatgpt(websocket: WebSocket, api_key: str):
    if OPENAI_API_KEY is None:
        raise Responses_400.not_supported_feature
    try:
        await websocket.accept()  # accept websocket
        try:  # get user from api key
            if api_key != OPENAI_API_KEY and not API_ENV == "test":
                api_key, user = await api_keys.get_api_key_and_owner(access_key=api_key)
            else:
                user: Users = Users(email=f"testaccount@{HOST_MAIN}")
        except Exception as exception:
            api_logger.error(exception, exc_info=True)
            return
        await begin_chat(
            websocket=websocket,
            user_id=user.email,
        )
    except WebSocketDisconnect:
        ...
    except Exception as exception:
        api_logger.error(exception, exc_info=True)
