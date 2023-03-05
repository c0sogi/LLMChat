from typing import List
from uuid import uuid4
from fastapi import APIRouter, Depends
from starlette.requests import Request
from app.common.config import MAX_API_KEY, MAX_API_WHITELIST
from app.database.schema import Users, ApiKeys, ApiWhiteLists, db, AsyncSession
from app import models as m
from app.errors import exceptions as ex
import string
import secrets
import ipaddress
from app.models import MessageOk, UserMe

router = APIRouter(prefix="/user")


@router.get("/me", response_model=UserMe)
async def get_me(request: Request):
    user = request.state.user
    user_info = await Users.one_or_nothing(id=user.id)
    return user_info


@router.put("/me")
async def put_me(request: Request):
    ...


@router.delete("/me")
async def delete_me(request: Request):
    ...


@router.get("/apikeys", response_model=List[m.GetApiKeyList])
async def get_api_keys(request: Request, session: AsyncSession = Depends(db.get_db)):
    """
    API KEY 조회
    """
    user = request.state.user
    api_keys = await ApiKeys.filter_by_condition(session, user_id=user.id)
    return api_keys.all()


@router.post("/apikeys", response_model=m.GetApiKeys)
async def create_api_keys(
    request: Request, key_info: m.AddApiKey, session: AsyncSession = Depends(db.get_db)
):
    """
    API KEY 생성
    """
    user = request.state.user
    api_keys = await ApiKeys.filter_by_condition(
        session, user_id=user.id, status="active"
    )
    if api_keys.count() == MAX_API_KEY:
        raise ex.MaxKeyCountEx()

    alphabet = string.ascii_letters + string.digits
    s_key = "".join(secrets.choice(alphabet) for _ in range(40))
    uid = None
    while not uid:
        uid_candidate = f"{str(uuid4())[:-12]}{str(uuid4())}"
        uid_check = await ApiKeys.one_or_nothing(access_key=uid_candidate)
        if not uid_check:
            uid = uid_candidate

    key_info = key_info.dict()
    new_api_key = await ApiKeys.create_new(
        session,
        auto_commit=True,
        secret_key=s_key,
        user_id=user.id,
        access_key=uid,
        **key_info,
    )
    session.commit()
    return new_api_key


@router.put("/apikeys/{key_id}", response_model=m.GetApiKeyList)
async def update_api_keys(
    request: Request,
    key_id: int,
    key_info: m.AddApiKey,
    session: AsyncSession = Depends(db.get_db),
):
    """
    API KEY User Memo Update
    """
    user = request.state.user
    key_datas = await ApiKeys.filter_by_condition(session, id=key_id)
    key_data = key_datas.one_or_none()
    if key_data is None:
        raise ex.NoKeyMatchEx()
    if key_data.user_id == user.id:
        return key_data.update(auto_commit=True, **key_info.dict())


@router.delete("/apikeys/{key_id}")
async def delete_api_keys(
    request: Request,
    key_id: int,
    access_key: str,
    session: AsyncSession = Depends(db.get_db),
):
    user = request.state.user
    await check_api_owner(user.id, key_id)
    search_by_key = await ApiKeys.filter_by_condition(session, access_key=access_key)
    if not search_by_key.first():
        raise ex.NoKeyMatchEx()
    search_by_key.delete(auto_commit=True)
    return MessageOk()


@router.get("/apikeys/{key_id}/whitelists", response_model=List[m.GetAPIWhiteLists])
async def get_api_keys_whitelists(
    request: Request, key_id: int, session: AsyncSession = Depends(db.get_db)
):
    user = request.state.user
    await check_api_owner(user.id, key_id)
    whitelists = await ApiWhiteLists.filter_by_condition(session, api_key_id=key_id)
    return whitelists.all()


@router.post("/apikeys/{key_id}/whitelists", response_model=m.GetAPIWhiteLists)
async def create_api_keys_whitelists(
    request: Request,
    key_id: int,
    ip: m.CreateAPIWhiteLists,
    session: AsyncSession = Depends(db.get_db),
):
    user = request.state.user
    await check_api_owner(user.id, key_id)

    try:
        ipaddress.ip_address(ip.ip_addr)
    except Exception as e:
        raise ex.InvalidIpEx(ip=ip.ip_addr, ex=e)
    if (
        await ApiWhiteLists.filter_by_condition(session, api_key_id=key_id)
    ).count() == MAX_API_WHITELIST:
        raise ex.MaxWLCountEx()
    ip_dup = await ApiWhiteLists.one_or_nothing(api_key_id=key_id, ip_addr=ip.ip_addr)
    if ip_dup:
        return ip_dup
    ip_reg = await ApiWhiteLists.create_new(
        session, auto_commit=True, api_key_id=key_id, ip_addr=ip.ip_addr
    )
    return ip_reg


@router.delete("/apikeys/{key_id}/whitelists/{list_id}")
async def delete_api_keys_whitelists(
    request: Request,
    key_id: int,
    list_id: int,
    session: AsyncSession = Depends(db.get_db),
):
    user = request.state.user
    await check_api_owner(user.id, key_id)
    api_whitelists = await ApiWhiteLists.filter_by_condition(
        session, id=list_id, api_key_id=key_id
    )
    api_whitelists.delete(auto_commit=True)
    return MessageOk()


async def check_api_owner(user_id, key_id):
    api_keys = await ApiKeys.one_or_nothing(id=key_id, user_id=user_id)
    if not api_keys:
        raise ex.NoKeyMatchEx()
