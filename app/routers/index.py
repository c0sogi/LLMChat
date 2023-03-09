# from datetime import datetime
from inspect import currentframe as frame
from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse  # Response
from fastapi.requests import Request

from app.errors.exceptions import APIException

# from app.background_tasks import background_task_state
# from sqlalchemy.orm import AsyncSession
# from fastapi import Depends
# from app.database.schema import Users

router = APIRouter(tags=["index"])


@router.get("/")
async def index():
    # now = f"UTC Time ({datetime.utcnow().strftime('%Y.%m.%d %H:%M:%S')})"
    # return Response(
    #     now,
    #     media_type="application/json; charset=utf8mb4",
    # )
    return RedirectResponse(url="/redoc")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("app/contents/favicon.ico")


@router.get("/test")
async def test(req: Request):
    """
    ELB status check
    """
    try:
        ...
    except Exception as exception:
        req.state.inspect = frame()
        raise APIException(ex=exception)
    else:
        return {"user_status": req.state.user}
