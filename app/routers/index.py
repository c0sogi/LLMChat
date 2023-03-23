import logging
from inspect import currentframe as frame
from fastapi import APIRouter, WebSocket
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from app.errors.exceptions import APIException, Responses_500
from app.utils.stream_manager import AsyncStream, establish_websocket_connection

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
    if request.url.scheme == "http":
        host_address: str = f"ws://{request.url.hostname}:{request.url.port}/ws/chatgpt/{user_id}/{api_key}"
    elif request.url.scheme == "https":
        host_address: str = (
            f"wss://{request.url.hostname}/ws/chatgpt/{user_id}/{api_key}"
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


@router.websocket("/ws/chatgpt/{user_id}/{api_key}")
async def ws_chatgpt(websocket: WebSocket, user_id: str, api_key: str):
    await websocket.accept()
    context: list[dict[str, str]] | None = AsyncStream.context.get(user_id)
    if context is None:
        context: list[dict[str, str]] = []
        AsyncStream.context[user_id] = context
    await establish_websocket_connection(
        websocket=websocket,
        user_id=user_id,
        api_key=api_key,
    )


@router.get("/test", status_code=200)
async def test(request: Request):
    try:
        result: StreamingResponse = StreamingResponse(
            content=AsyncStream.hello_world(),
            headers={"Content-Type": "text/event-stream; charset=UTF-8"},
        )
    except Exception as exception:
        logging.error(exception)
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
