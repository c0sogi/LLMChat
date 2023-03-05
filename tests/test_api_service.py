from tests.conftest import *
from requests import Response, get
from app.utils.date_utils import UTC
from app.utils.encoding_and_hashing import hash_params
from app.utils.query_utils import parse_params


def test_request_api(login_header, client):
    def get_apikey(client, login_header):
        res = client.post(
            "api/user/apikeys", json={"user_memo": "user1__key"}, headers=login_header
        )
        res_body = res.json()
        assert res.status_code == 200
        assert "access_key" in res_body
        assert "secret_key" in res_body
        apikey = {
            "access_key": res_body["access_key"],
            "secret_key": res_body["secret_key"],
        }

        res = client.get("api/user/apikeys", headers=login_header)
        res_body = res.json()
        assert res.status_code == 200
        assert "user1__key" in res_body[0]["user_memo"]
        return apikey

    apikey = get_apikey(client=client, login_header=login_header)
    parsed_qs: str = parse_params(
        params={"key": apikey["access_key"], "timestamp": UTC.timestamp(hour_diff=9)}
    )
    res = client.get(
        f"/api/services?{parsed_qs}",
        headers={"secret": hash_params(qs=parsed_qs, secret_key=apikey["secret_key"])},
    )
    assert res.status_code == 200
