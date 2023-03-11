from tests.conftest import *
import pytest
from time import sleep
from app.utils.date_utils import UTC
from app.utils.encoding_and_hashing import hash_params
from app.utils.query_utils import parse_params


@pytest.mark.asyncio
async def test_request_api(login_header, client):
    async def _get_apikey(client, authorized_header):
        res = await client.post(
            "api/user/apikeys",
            json={"user_memo": "user1__key"},
            headers=authorized_header,
        )
        res_body = res.json()
        assert res.status_code == 200
        assert "access_key" in res_body
        assert "secret_key" in res_body
        apikey = {
            "access_key": res_body["access_key"],
            "secret_key": res_body["secret_key"],
        }

        res = await client.get("api/user/apikeys", headers=authorized_header)
        res_body = res.json()
        assert res.status_code == 200
        assert "user1__key" in res_body[0]["user_memo"]
        return apikey

    apikey = await _get_apikey(client=client, authorized_header=login_header)
    parsed_qs: str = parse_params(
        params={"key": apikey["access_key"], "timestamp": UTC.timestamp(hour_diff=9)}
    )
    sleep(5)
    res = await client.get(
        f"/api/services?{parsed_qs}",
        headers={"secret": hash_params(qs=parsed_qs, secret_key=apikey["secret_key"])},
    )
    assert res.status_code in (200, 307)
