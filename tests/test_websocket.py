from fastapi.testclient import TestClient
import pytest
from starlette.testclient import WebSocketTestSession
from app.common.config import OPENAI_API_KEY
from app.viewmodels.base_models import ReceiveChatMessage, SendChatMessage


@pytest.mark.asyncio
async def test_chatgpt(fake_client: TestClient, base_websocket_url: str):
    with fake_client.websocket_connect(f"{base_websocket_url}/ws/chatgpt/{OPENAI_API_KEY}") as ws_client:
        assert isinstance(ws_client, WebSocketTestSession)
        # send message to websocket
        ws_client.send_text(ReceiveChatMessage(msg="/ping", translate=False, chat_room_id=0).json())
        # receive message from websocket
        client_received: SendChatMessage = SendChatMessage.parse_raw(ws_client.receive_text())
        assert client_received.msg == "pong"
        # close websocket
        ws_client.close()
