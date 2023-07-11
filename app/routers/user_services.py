from typing import Optional

from fastapi import APIRouter

from app.models.base_models import MessageOk

router = APIRouter(prefix="/user-services")


@router.get("/", status_code=200)
def test(query: Optional[str] = None):
    return MessageOk(message="Hello, World!" if query is None else query)
