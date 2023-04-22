import time
from fastapi.testclient import TestClient
import logging
import pytest
from starlette.testclient import WebSocketTestSession
from app.common.config import OPENAI_API_KEY, Config
from app.viewmodels.base_models import MessageFromWebsocket, MessageToWebsocket
from app.utils.chatgpt.chatgpt_cache_manager import chatgpt_cache_manager
from app.viewmodels.gpt_models import MessageHistory, UserGptContext

chatgpt_cache_manager.cache.start(config=Config.get(option="test"))


@pytest.mark.skip
@pytest.mark.asyncio
async def test_chatgpt_redis():
    # set random context
    user_id: str = ""
    role: str = "user"
    message: str = "test message"
    default_context: UserGptContext = UserGptContext.construct_default(user_id=user_id)

    # reset context
    await chatgpt_cache_manager.reset_context(user_id=user_id)

    # create context
    await chatgpt_cache_manager.create_context(
        user_id="",
        user_gpt_context=default_context,
    )
    # read context
    context: UserGptContext = await chatgpt_cache_manager.read_context(user_id="")
    assert context == default_context

    # add new message to redis
    new_message: MessageHistory = MessageHistory(
        role=role, content=message, is_user=True, tokens=len(context.tokenize(message))
    )
    await chatgpt_cache_manager.append_message_history(user_id=user_id, role=role, message_history=new_message)

    # read message from redis
    message_histories: list[MessageHistory] = await chatgpt_cache_manager.get_message_history(
        user_id=user_id, role=role
    )
    assert message_histories == [new_message]


@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OpenAI API Key is not set")
def test_chatgpt_connection(websocket_app: TestClient, base_websocket_url: str):
    with websocket_app.websocket_connect(f"{base_websocket_url}/ws/chatgpt/{OPENAI_API_KEY}") as ws_client:
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


@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OpenAI API Key is not set")
def test_chatgpt_conversation(websocket_app: TestClient, base_websocket_url: str):
    # parameters
    timeout: int = 10
    with websocket_app.websocket_connect(f"{base_websocket_url}/ws/chatgpt/{OPENAI_API_KEY}") as ws_client:
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
