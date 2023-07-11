import ipaddress

from fastapi import APIRouter
from starlette.requests import Request

from app.database.crud import api_keys, api_whitelists, users
from app.errors.api_exceptions import Responses_400
from app.models.base_models import (
    AddApiKey,
    CreateApiWhiteList,
    GetApiKey,
    GetApiKeyFirstTime,
    GetApiWhiteList,
    MessageOk,
    UserMe,
)

router = APIRouter(prefix="/user")


@router.get("/me", response_model=UserMe)
async def get_me(request: Request):
    return await users.get_me(user_id=request.state.user.id)


@router.put("/me")
async def put_me(request: Request):
    ...


@router.delete("/me")
async def delete_me(request: Request):
    ...


@router.get("/apikeys", response_model=list[GetApiKey])
async def get_api_keys(request: Request):
    return await api_keys.get_api_keys(user_id=request.state.user.id)


@router.post("/apikeys", response_model=GetApiKeyFirstTime, status_code=201)
async def create_api_key(request: Request, api_key_info: AddApiKey):
    return await api_keys.create_api_key(
        user_id=request.state.user.id, additional_key_info=api_key_info
    )


@router.put("/apikeys/{key_id}", response_model=GetApiKey)
async def update_api_key(
    request: Request,
    api_key_id: int,
    api_key_info: AddApiKey,
):
    """
    API KEY User Memo Update
    """
    return await api_keys.update_api_key(
        updated_key_info=api_key_info.dict(),
        access_key_id=api_key_id,
        user_id=request.state.user.id,
    )


@router.delete("/apikeys/{key_id}")
async def delete_api_key(
    request: Request,
    key_id: int,
    access_key: str,
):
    await api_keys.delete_api_key(
        access_key_id=key_id,
        access_key=access_key,
        user_id=request.state.user.id,
    )
    return MessageOk()


@router.get(
    "/apikeys/{key_id}/whitelists", response_model=list[GetApiWhiteList]
)
async def get_api_keys_whitelist(api_key_id: int):
    return await api_whitelists.get_api_key_whitelist(api_key_id=api_key_id)


@router.post("/apikeys/{key_id}/whitelists", response_model=GetApiWhiteList)
async def create_api_keys_whitelist(
    api_key_id: int,
    ip: CreateApiWhiteList,
):
    ip_address: str = ip.ip_address
    try:
        ipaddress.ip_address(ip_address)
    except Exception as exception:
        raise Responses_400.invalid_ip(
            lazy_format={"ip": ip_address}, ex=exception
        )
    return await api_whitelists.create_api_key_whitelist(
        ip_address=ip_address, api_key_id=api_key_id
    )


@router.delete("/apikeys/{key_id}/whitelists/{list_id}")
async def delete_api_keys_whitelist(
    request: Request,
    api_key_id: int,
    whitelist_id: int,
):
    await api_whitelists.delete_api_key_whitelist(
        user_id=request.state.user.id,
        api_key_id=api_key_id,
        whitelist_id=whitelist_id,
    )
    return MessageOk()
