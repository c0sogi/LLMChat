from typing import AsyncGenerator, AsyncIterator, Generator, Iterator
from fastapi import WebSocket

from app.utils.chatgpt.chatgpt_buffer import BufferedUserContext
from app.utils.chatgpt.chatgpt_generation import (
    message_history_organizer,
)
from app.viewmodels.base_models import InitMessage, MessageToWebsocket


class SendToWebsocket:
    @staticmethod
    async def init(
        buffer: BufferedUserContext,
        send_chat_rooms: bool = True,
        send_previous_chats: bool = True,
        init_callback: bool = True,
    ) -> None:
        """Send initial message to websocket, providing current state of user"""
        previous_chats = message_history_organizer(
            user_gpt_context=buffer.current_user_gpt_context,
            send_to_stream=False,
        )
        assert isinstance(previous_chats, list)
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=InitMessage(
                chat_rooms=buffer.sorted_chat_rooms if send_chat_rooms else None,
                previous_chats=previous_chats if send_previous_chats else None,
                init_callback=init_callback,
            ).json(),
            chat_room_id=buffer.current_chat_room_id,
            init=True,
        )

    @staticmethod
    async def message(
        websocket: WebSocket,
        msg: str,
        chat_room_id: str,
        finish: bool = True,
        is_user: bool = False,
        init: bool = False,
        model_name: str | None = None,
    ) -> None:
        """Send whole message to websocket"""
        await websocket.send_json(  # send stream message
            MessageToWebsocket(
                msg=msg,
                finish=finish,
                chat_room_id=chat_room_id,
                is_user=is_user,
                init=init,
                model_name=model_name,
            ).dict()
        )

    @staticmethod
    async def stream(
        buffer: BufferedUserContext,
        stream: AsyncGenerator | Generator | AsyncIterator | Iterator,
        finish: bool = True,
        is_user: bool = False,
        chunk_size: int = 2,
        model_name: str | None = None,
    ) -> str:
        """Send SSE stream to websocket"""
        final_response, stream_buffer = "", ""
        iteration: int = 0
        await buffer.websocket.send_json(
            MessageToWebsocket(
                msg=None,
                finish=False,
                chat_room_id=buffer.current_chat_room_id,
                is_user=is_user,
                model_name=model_name,
            ).dict()
        )
        try:
            if isinstance(stream, (Generator, Iterator)):
                for delta in stream:  # stream from local
                    if buffer.done.is_set():
                        buffer.done.clear()
                        raise InterruptedError("Stream was interrupted by user.")
                    stream_buffer += delta
                    iteration += 1
                    if iteration % chunk_size == 0:
                        final_response += stream_buffer
                        await buffer.websocket.send_json(
                            MessageToWebsocket(
                                msg=stream_buffer,
                                finish=False,
                                chat_room_id=buffer.current_chat_room_id,
                                is_user=is_user,
                            ).dict()
                        )
                        stream_buffer = ""
            elif isinstance(stream, (AsyncGenerator, AsyncIterator)):
                async for delta in stream:  # stream from api
                    if buffer.done.is_set():
                        buffer.done.clear()
                        raise InterruptedError("Stream was interrupted by user.")
                    stream_buffer += delta
                    iteration += 1
                    if iteration % chunk_size == 0:
                        final_response += stream_buffer
                        await buffer.websocket.send_json(
                            MessageToWebsocket(
                                msg=stream_buffer,
                                finish=False,
                                chat_room_id=buffer.current_chat_room_id,
                                is_user=is_user,
                            ).dict()
                        )
                        stream_buffer = ""
            else:
                raise TypeError("Stream type is not AsyncGenerator or Generator.")
        except InterruptedError as e:
            await buffer.websocket.send_json(
                MessageToWebsocket(
                    msg=stream_buffer,
                    finish=True,
                    chat_room_id=buffer.current_chat_room_id,
                    is_user=is_user,
                ).dict()
            )
            raise e
        await buffer.websocket.send_json(
            MessageToWebsocket(
                msg=stream_buffer,
                finish=True if finish else False,
                chat_room_id=buffer.current_chat_room_id,
                is_user=is_user,
            ).dict()
        )
        return final_response
