import json
import time

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from app.common.config import OPENAI_API_KEY
from app.models.chat_models import (
    ChatRoles,
    MessageHistory,
    UserChatContext,
    UserChatProfile,
)
from app.models.llms import LLMModels
from app.models.base_models import MessageFromWebsocket, MessageToWebsocket


# @pytest.mark.skip
@pytest.mark.asyncio
async def test_chat_redis(cache_manager):
    # set random context
    user_id: str = "test_user"
    test_chat_room_id: str = "test_chat_room"
    role: ChatRoles = ChatRoles.USER
    message: str = "test message"
    default_context: UserChatContext = UserChatContext.construct_default(
        user_id=user_id, chat_room_id=test_chat_room_id
    )
    new_context: UserChatContext = UserChatContext(
        user_chat_profile=UserChatProfile(
            user_id=user_id,
            chat_room_id=test_chat_room_id,
        ),
        llm_model=LLMModels.gpt_4,
    )

    # delete test chat room
    await cache_manager.delete_chat_room(
        user_id=user_id, chat_room_id=test_chat_room_id
    )

    # create new context
    await cache_manager.create_context(
        user_chat_context=new_context,
    )
    # read new context
    assert (
        new_context.user_chat_profile.chat_room_id
        == (
            await cache_manager.read_context_from_profile(
                user_chat_profile=new_context.user_chat_profile
            )
        ).user_chat_profile.chat_room_id
    )

    # add new message to redis
    new_message: MessageHistory = MessageHistory(
        role=role.value,
        content=message,
        actual_role=role.value,
        tokens=new_context.get_tokens_of(message),
    )
    await cache_manager.append_message_history(
        user_id=user_id,
        chat_room_id=test_chat_room_id,
        role=role,
        message_history=new_message,
    )

    # read message from redis
    message_histories: list[MessageHistory] = await cache_manager.get_message_history(
        user_id=user_id, chat_room_id=test_chat_room_id, role=role
    )
    assert message_histories == [new_message]

    # reset context and read context
    await cache_manager.reset_context(user_id=user_id, chat_room_id=test_chat_room_id)
    assert (
        default_context.user_chat_profile.chat_room_id
        == (
            await cache_manager.read_context_from_profile(
                user_chat_profile=default_context.user_chat_profile
            )
        ).user_chat_profile.chat_room_id
    )

    # delete test chat room
    await cache_manager.delete_chat_room(
        user_id=user_id, chat_room_id=test_chat_room_id
    )


@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OpenAI API Key is not set")
def test_chat_connection(client: TestClient, base_websocket_url: str):
    with client.websocket_connect(
        f"{base_websocket_url}/ws/chat/{OPENAI_API_KEY}"
    ) as ws_client:
        assert isinstance(ws_client, WebSocketTestSession)

        client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(
            ws_client.receive_text()
        )
        assert client_received.init
        client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(
            ws_client.receive_text()
        )
        assert client_received.msg is not None and "tokens" in json.loads(
            client_received.msg
        )
        assert client_received.chat_room_id is not None
        # send message to websocket
        ws_client.send_json(
            MessageFromWebsocket(
                msg="/ping",
                chat_room_id=client_received.chat_room_id,
            ).dict()
        )
        # receive message from websocket
        client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(
            ws_client.receive_text()
        )
        assert client_received.msg == "pong"
        # close websocket
        ws_client.close()


@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OpenAI API Key is not set")
def test_chat_conversation(client: TestClient, base_websocket_url: str, test_logger):
    # parameters
    timeout: int = 10
    with client.websocket_connect(
        f"{base_websocket_url}/ws/chat/{OPENAI_API_KEY}"
    ) as ws_client:
        assert isinstance(ws_client, WebSocketTestSession)
        client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(
            ws_client.receive_text()
        )
        assert client_received.init

        assert client_received.chat_room_id is not None
        # send message to websocket
        ws_client.send_json(
            MessageFromWebsocket(
                msg="say this word: TEST",
                translate=None,
                chat_room_id=client_received.chat_room_id,
            ).dict()
        )

        # receive messages from websocket, loop until received message with finish=True
        # timeout: 10 seconds
        received_messages: list[MessageToWebsocket] = []
        now: float = time.time()
        while time.time() - now < timeout:
            client_received: MessageToWebsocket = MessageToWebsocket.parse_raw(
                ws_client.receive_text()
            )
            received_messages.append(client_received)
            if client_received.finish:
                break
        assert len(received_messages) > 0

        # show received messages
        for msg in received_messages:
            test_logger.info(msg)

        # assemble msg from received messages using list comprehension
        received_msg: str = "".join(
            [msg.msg for msg in received_messages if msg.msg is not None]
        )
        assert "TEST" in received_msg

        # close websocket
        ws_client.close()
