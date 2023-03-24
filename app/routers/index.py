from uuid import uuid4
from inspect import currentframe as frame
from fastapi import APIRouter, WebSocket
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
)
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from app.common.config import OPENAI_API_KEY
from app.errors.exceptions import APIException, Responses_500
from app.utils.logger import logger
from app.utils.stream_manager import (
    ChatGptStreamManager,
    UserGptContext,
    UserGptProfile,
    GPT_MODELS,
)

router = APIRouter(tags=["index"])
templates = Jinja2Templates(directory="app")


@router.get("/")
async def index():
    return RedirectResponse("/docs", status_code=307)


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("app/contents/favicon.ico")


@router.get("/chatgpt/{user_id}/{api_key}")
async def chatgpt(request: Request, user_id: str, api_key: str):
    random_uuid: str = str(uuid4())
    if request.url.scheme == "http":
        host_address: str = f"ws://{request.url.hostname}:{request.url.port}/ws/chatgpt/{random_uuid}/{api_key}"
    elif request.url.scheme == "https":
        host_address: str = (
            f"wss://{request.url.hostname}/ws/chatgpt/{random_uuid}/{api_key}"
        )
    else:
        raise Responses_500.websocket_error
    return templates.TemplateResponse(
        name="chatgpt.html",
        context={
            "request": request,
            "host_address": host_address,
            "user_id": user_id,
        },
    )


@router.websocket("/ws/chatgpt/{random_uuid}/{api_key}")
async def ws_chatgpt(websocket: WebSocket, random_uuid: str, api_key: str):
    user_gpt_context: UserGptContext | None = (
        ChatGptStreamManager.user_gpt_contexts.get(random_uuid)
    )
    try:
        await websocket.accept()  # accept websocket
        if user_gpt_context is None:
            user_gpt_context: UserGptContext = UserGptContext(
                gpt_model=GPT_MODELS.gpt_3_5_turbo,
                user_gpt_profile=UserGptProfile(user_id=random_uuid),
            )
            ChatGptStreamManager.user_gpt_contexts[random_uuid] = user_gpt_context
        if user_gpt_context.is_user_in_chat:
            ChatGptStreamManager.send_whole_message(
                websocket=websocket, message="이미 같은 아이디로 채팅방에 입장되어 있습니다."
            )
            return
        await ChatGptStreamManager.send_websocket(
            websocket=websocket,
            user_gpt_context=user_gpt_context,
            api_key=api_key,
        )
    except Exception as exception:
        logger.error(exception)


@router.get("/test", status_code=200)
async def test(request: Request):
    try:
        user_id: str = "test_user"
        if request.url.scheme == "http":
            host_address: str = f"ws://{request.url.hostname}:{request.url.port}/ws/chatgpt/{user_id}/{OPENAI_API_KEY}"
        elif request.url.scheme == "https":
            host_address: str = (
                f"wss://{request.url.hostname}/ws/chatgpt/{user_id}/{OPENAI_API_KEY}"
            )
        else:
            raise Responses_500.websocket_error
        result = templates.TemplateResponse(
            name="chatgpt.html",
            context={
                "request": request,
                "host_address": host_address,
                "user_id": user_id,
            },
        )
    except Exception as exception:
        logger.error(exception)
        request.state.inspect = frame()
        raise APIException(ex=exception)
    else:
        return result


# messages: list = []
# @router.get("/chatgpt", status_code=200, response_class=StreamingResponse)
# async def get_chatgpt_endpoint():
#     messages.append(
#         {
#             "role": "user",
#             "content": "내가 방금 뭐라고 말했지?",
#         }
#     )

#     return StreamingResponse(
#         content=TextStream.chatgpt(context=messages),
#         headers={"Content-Type": "text/event-stream; charset=UTF-8"},
#     )


# @router.post("/chatgpt", status_code=200, response_class=StreamingResponse)
# async def post_chatgpt_endpoint(user_chat_gpt_message: UserChatGptMessage):
#     messages.append(
#         {
#             "role": "user",
#             "content": user_chat_gpt_message.user_message,
#         }
#     )
#     return StreamingResponse(
#         content=TextStream.chatgpt(context=messages),
#         headers={"Content-Type": "text/event-stream; charset=UTF-8"},
#     )
