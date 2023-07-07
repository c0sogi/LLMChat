from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/")
async def index():
    return FileResponse("app/web/index.html")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("app/contents/favicon.ico")


@router.get("/test")
async def test(request: Request):
    return {"username": request.session.get("username", None)}
