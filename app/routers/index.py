from inspect import currentframe as frame
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from app.common.config import OPENAI_API_KEY
from app.utils.logger import api_logger

router = APIRouter(redirect_slashes=True)
templates = Jinja2Templates(directory="app")


@router.get("/")
async def index():
    return FileResponse("app/web/index.html")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("app/contents/favicon.ico")


@router.get("/test", status_code=200)
async def test(request: Request):
    try:
        if request.url.scheme == "http":
            host_address: str = f"ws://{request.url.hostname}:{request.url.port}/ws/chatgpt/{OPENAI_API_KEY}"
        elif request.url.scheme == "https":
            host_address: str = f"wss://{request.url.hostname}/ws/chatgpt/{OPENAI_API_KEY}"
        else:
            api_logger.error("index error")
        return templates.TemplateResponse(
            status_code=200,
            name="chatgpt.html",
            context={
                "request": request,
                "host_address": host_address,
            },
        )
    except Exception as exception:
        api_logger.error(exception)
        request.state.inspect = frame()
        return JSONResponse({"detail": "Internal server error"}, status_code=500)
