from fastapi import APIRouter, WebSocket
from app.common.config import HOST_MAIN, OPENAI_API_KEY
from app.database.crud import get_api_key_owner
from app.database.schema import Users
from app.models.gpt_models import GPT_MODELS, UserGptContext, UserGptProfile
from app.utils.chatgpt.chatgpt_config import GPT_CONFIG
from app.utils.logger import logger
from app.utils.chatgpt.chatgpt_stream_manager import begin_chat, SendToWebsocket

router = APIRouter()


@router.websocket("/chatgpt/{api_key}")
async def ws_chatgpt(websocket: WebSocket, api_key: str):
    try:
        await websocket.accept()  # accept websocket
        try:
            if api_key != OPENAI_API_KEY:
                user: Users = await get_api_key_owner(access_key=api_key)
            else:
                user: Users = Users(email=f"testaccount@{HOST_MAIN}")
        except Exception:
            await SendToWebsocket.message(websocket=websocket, msg="유효하지 않은 API 키입니다. 연결을 종료합니다.")
            return
        # create new user gpt context if not exist
        user_gpt_context: UserGptContext | None = GPT_CONFIG.user_gpt_contexts.get(user.email, None)
        if user_gpt_context is None:
            user_gpt_context: UserGptContext = UserGptContext(
                gpt_model=GPT_MODELS.gpt_3_5_turbo,
                user_gpt_profile=UserGptProfile(user_id=user.email),
            )
            GPT_CONFIG.user_gpt_contexts[user.email] = user_gpt_context
        # while user_gpt_context.is_user_in_chat:
        #     await ChatGptStreamManager.send_whole_message(websocket=websocket, msg="이미 같은 아이디로 채팅방에 입장되어 있습니다.")
        #     await websocket.receive_text()
        await begin_chat(
            websocket=websocket,
            user_gpt_context=user_gpt_context,
            openai_api_key=OPENAI_API_KEY,
        )
    except Exception as exception:
        logger.error(exception)
