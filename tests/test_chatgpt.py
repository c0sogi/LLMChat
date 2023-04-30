import time
from fastapi.testclient import TestClient
import pytest
from starlette.testclient import WebSocketTestSession
from app.common.config import HOST_MAIN, OPENAI_API_KEY, Config
from app.viewmodels.base_models import MessageFromWebsocket, MessageToWebsocket
from app.viewmodels.gpt_models import GPT_MODELS, GptRoles, MessageHistory, UserGptContext, UserGptProfile


# @pytest.mark.skip
@pytest.mark.asyncio
async def test_chatgpt_redis(chatgpt_cache_manager):
    # set random context
    user_id: str = "test_user"
    test_chat_room_id: str = "test_chat_room"
    role: str = GptRoles.USER
    message: str = "test message"
    default_context: UserGptContext = UserGptContext.construct_default(user_id=user_id, chat_room_id=test_chat_room_id)
    new_context: UserGptContext = UserGptContext(
        user_gpt_profile=UserGptProfile(user_id=user_id, chat_room_id=test_chat_room_id, user_role="test_user"),
        gpt_model=GPT_MODELS.gpt_4,
    )

    # delete test chat room
    await chatgpt_cache_manager.delete_chat_room(user_id=user_id, chat_room_id=test_chat_room_id)

    # create new context
    await chatgpt_cache_manager.create_context(
        user_gpt_context=new_context,
    )
    # read new context
    assert new_context == await chatgpt_cache_manager.read_context(user_id=user_id, chat_room_id=test_chat_room_id)

    # add new message to redis
    new_message: MessageHistory = MessageHistory(
        role=role, content=message, is_user=True, tokens=len(new_context.tokenize(message))
    )
    await chatgpt_cache_manager.append_message_history(
        user_id=user_id, chat_room_id=test_chat_room_id, role=role, message_history=new_message
    )

    # read message from redis
    message_histories: list[MessageHistory] = await chatgpt_cache_manager.get_message_history(
        user_id=user_id, chat_room_id=test_chat_room_id, role=role
    )
    assert message_histories == [new_message]

    # reset context and read context
    await chatgpt_cache_manager.reset_context(user_id=user_id, chat_room_id=test_chat_room_id)
    assert default_context == await chatgpt_cache_manager.read_context(user_id=user_id, chat_room_id=test_chat_room_id)

    # delete test chat room
    await chatgpt_cache_manager.delete_chat_room(user_id=user_id, chat_room_id=test_chat_room_id)


@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OpenAI API Key is not set")
def test_chatgpt_connection(websocket_app: TestClient, base_websocket_url: str):
    with websocket_app.websocket_connect(f"{base_websocket_url}/ws/chatgpt/{OPENAI_API_KEY}") as ws_client:
        assert isinstance(ws_client, WebSocketTestSession)

        client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(ws_client.receive_text())
        assert client_received.init
        # send message to websocket
        ws_client.send_text(
            MessageFromWebsocket(msg="/ping", translate=False, chat_room_id=client_received.chat_room_id).json()
        )
        # receive message from websocket
        client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(ws_client.receive_text())
        assert client_received.msg == "pong"
        # close websocket
        ws_client.close()


@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OpenAI API Key is not set")
def test_chatgpt_conversation(websocket_app: TestClient, base_websocket_url: str, test_logger):
    # parameters
    timeout: int = 10
    with websocket_app.websocket_connect(f"{base_websocket_url}/ws/chatgpt/{OPENAI_API_KEY}") as ws_client:
        assert isinstance(ws_client, WebSocketTestSession)
        client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(ws_client.receive_text())
        assert client_received.init

        # send message to websocket
        ws_client.send_text(
            MessageFromWebsocket(
                msg="say this word: TEST", translate=False, chat_room_id=client_received.chat_room_id
            ).json()
        )

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
            test_logger.info(msg)

        # assemble msg from received messages using list comprehension
        received_msg: str = "".join([msg.msg for msg in received_messages])
        assert "TEST" in received_msg

        # close websocket
        ws_client.close()
