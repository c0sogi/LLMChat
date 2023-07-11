from fastapi import WebSocket

from app.models.chat_models import (
    ResponseType,
    UserChatContext,
    command_response,
)
from app.models.llms import LLMModels
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.websocket import SendToWebsocket


class TestingCommands:
    @staticmethod
    @command_response.send_message_and_stop
    def test(
        user_chat_context: UserChatContext,
    ) -> str:  # test command showing user_chat_context
        """Test command showing user_chat_context\n
        /test"""
        return str(user_chat_context)

    @staticmethod
    @command_response.send_message_and_stop
    def ping() -> str:
        """Ping! Pong!\n
        /ping"""
        return "pong"

    @staticmethod
    @command_response.send_message_and_stop
    def echo(msg: str, /) -> str:
        """Echo your message\n
        /echo <msg>"""
        return msg

    @staticmethod
    @command_response.do_nothing
    async def sendtowebsocket(
        msg: str, /, websocket: WebSocket, user_chat_context: UserChatContext
    ) -> None:
        """Send all messages to websocket\n
        /sendtowebsocket"""
        await SendToWebsocket.message(
            websocket=websocket,
            msg=msg,
            chat_room_id=user_chat_context.chat_room_id,
        )

    @staticmethod
    async def testchaining(
        chain_size: int, buffer: BufferedUserContext
    ) -> tuple[str, ResponseType]:
        """Test chains of commands\n
        /testchaining <size_of_chain>"""
        if chain_size <= 0:
            return "Chaining Complete!", ResponseType.SEND_MESSAGE_AND_STOP
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=f"Current Chaining: {chain_size}",
            chat_room_id=buffer.current_chat_room_id,
        )
        return f"/testchaining {chain_size-1}", ResponseType.REPEAT_COMMAND

    @staticmethod
    @command_response.send_message_and_stop
    def codeblock(language, codes: str, /) -> str:
        """Send codeblock\n
        /codeblock <language> <codes>"""
        return f"\n```{language.lower()}\n" + codes + "\n```\n"

    @staticmethod
    @command_response.send_message_and_stop
    def addmember(name: str, /, user_chat_context: UserChatContext) -> str:
        LLMModels.add_member(name, None)
        return f"Added {name} to members"
