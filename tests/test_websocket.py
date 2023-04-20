import time
from fastapi.testclient import TestClient
import pytest
import logging
from starlette.testclient import WebSocketTestSession
from app.common.config import HOST_MAIN, OPENAI_API_KEY
from app.utils.chatgpt.chatgpt_context_manager import chatgpt_cache_manager
from app.viewmodels.base_models import MessageFromWebsocket, MessageToWebsocket


@pytest.mark.asyncio
async def test_chatgpt_connection(fake_client: TestClient, base_websocket_url: str):
    with fake_client.websocket_connect(f"{base_websocket_url}/ws/chatgpt/{OPENAI_API_KEY}") as ws_client:
        assert isinstance(ws_client, WebSocketTestSession)
        client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(ws_client.receive_text())
        assert client_received.init
        # send message to websocket
        ws_client.send_text(MessageFromWebsocket(msg="/ping", translate=False, chat_room_id=0).json())
        # receive message from websocket
        client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(ws_client.receive_text())
        assert client_received.msg == "pong"
        # close websocket
        ws_client.close()


@pytest.mark.asyncio
async def test_chatgpt_conversation(fake_client: TestClient, base_websocket_url: str):
    # parameters
    timeout: int = 10
    # clear redis cache
    chatgpt_cache_manager.reset_context(user_id=f"testaccount@{HOST_MAIN}")
    with fake_client.websocket_connect(f"{base_websocket_url}/ws/chatgpt/{OPENAI_API_KEY}") as ws_client:
        assert isinstance(ws_client, WebSocketTestSession)
        client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(ws_client.receive_text())
        assert client_received.init

        # send message to websocket
        ws_client.send_text(MessageFromWebsocket(msg="say this word: TEST", translate=False, chat_room_id=0).json())

        # receive messages from websocket, loop until received message with finish=True
        # timeout: 10 seconds
        received_messages: list[MessageToWebsocket] = []
        now: float = time.time()
        while time.time() - now < timeout:
            client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(ws_client.receive_text())
            received_messages.append(client_received)
            if client_received.finish:
                break
        assert len(received_messages) > 1

        # show received messages
        for msg in received_messages:
            logging.critical(msg)

        # assemble msg from received messages using list comprehension
        received_msg: str = "".join([msg.msg for msg in received_messages])
        assert "TEST" in received_msg

        # close websocket
        ws_client.close()
