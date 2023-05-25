import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Generator, Iterator, Optional

from fastapi import WebSocket

from app.errors.chat_exceptions import ChatLengthException, ChatModelNotImplementedException
from app.models.llms import LLMModels
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.chat_config import ChatConfig
from app.utils.chat.prompts import message_history_organizer
from app.utils.logger import api_logger
from app.viewmodels.base_models import InitMessage, MessageToWebsocket, StreamProgress


class SyncToAsyncGenerator:
    def __init__(self, sync_gen: Iterator):
        self._gen: Iterator[Any] = sync_gen
        self._queue: asyncio.Queue = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self._queue.get()
        if value is self:  # using `self` as sentinel
            raise StopAsyncIteration
        return value

    async def run(self, executor: Optional[ThreadPoolExecutor] = None):
        it = iter(self._gen)  # Get an iterator from the generator
        while True:
            try:
                # Use 'self' as a sentinel value to avoid raising StopIteration
                value = await asyncio.get_running_loop().run_in_executor(executor, next, it, self)
                if value is self:  # Check if the iterator is exhausted
                    await self._queue.put(self)  # Notify of completion
                    break
                else:
                    await self._queue.put(value)
            except StopIteration:
                # This exception is expected and handled gracefully
                break
            except Exception as e:
                # Other exceptions are unexpected and should propagate up
                api_logger.exception(f"Unexpected exception in SyncToAsyncGenerator: {e}")
                raise e


class SendToWebsocket:
    @staticmethod
    async def init(
        buffer: BufferedUserContext,
        send_chat_rooms: bool = False,
        send_previous_chats: bool = False,
        send_models: bool = False,
        send_selected_model: bool = False,
        send_tokens: bool = False,
        wait_next_query: bool = False,
    ) -> None:
        """Send initial message to websocket, providing current state of user"""
        previous_chats = message_history_organizer(
            user_chat_context=buffer.current_user_chat_context,
            send_to_stream=False,
        )
        assert isinstance(previous_chats, list)
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=InitMessage(
                chat_rooms=buffer.sorted_chat_rooms if send_chat_rooms else None,
                previous_chats=previous_chats if send_previous_chats else None,
                models=LLMModels._member_names_ if send_models else None,
                selected_model=buffer.current_user_chat_context.llm_model.name
                if send_selected_model or send_previous_chats or send_models
                else None,
                tokens=buffer.current_user_chat_context.total_tokens if send_tokens else None,
                wait_next_query=wait_next_query,
            ).json(),
            chat_room_id=buffer.current_chat_room_id,
            init=True,
            finish=False,
        )

    @staticmethod
    async def message(
        websocket: WebSocket,
        chat_room_id: str,
        finish: bool = True,
        is_user: bool = False,
        init: bool = False,
        msg: Optional[str] = None,
        model_name: Optional[str] = None,
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

    @classmethod
    async def stream(
        cls,
        buffer: BufferedUserContext,
        stream_func: Callable,
        stream_kwargs: dict,
        stream_progress: StreamProgress,
        finish: bool = True,
        is_user: bool = False,
        chunk_size: int = 1,
        model_name: Optional[str] = None,
    ) -> None:
        """Send SSE stream to websocket"""

        async def hand_shake() -> None:
            # Send initial message
            await cls.message(
                websocket=buffer.websocket,
                msg=None,
                chat_room_id=buffer.current_chat_room_id,
                finish=False,
                is_user=is_user,
                model_name=model_name,
            )

        async def consumer(async_stream: AsyncIterator) -> None:
            """Helper function to send chunks of data"""
            api_logger.info("Sending stream to websocket")
            iteration: int = 0
            async for delta in async_stream:  # stream from api
                if buffer.done.is_set():
                    raise InterruptedError(stream_progress.response + stream_progress.buffer)
                stream_progress.buffer += delta
                iteration += 1
                if iteration % chunk_size == 0:
                    stream_progress.response += stream_progress.buffer
                    await cls.message(
                        websocket=buffer.websocket,
                        msg=stream_progress.buffer,
                        chat_room_id=buffer.current_chat_room_id,
                        finish=False,
                        is_user=is_user,
                        model_name=model_name,
                    )
                    stream_progress.buffer = ""

        async def transmission() -> None:
            try:
                # Use Python 3.10's pattern matching for cleaner type checking
                match stream_func(**stream_kwargs):
                    case _stream if isinstance(_stream, (AsyncGenerator, AsyncIterator)):
                        await consumer(async_stream=_stream)

                    case _stream if isinstance(_stream, (Generator, Iterator)):
                        async_stream = SyncToAsyncGenerator(sync_gen=_stream)
                        await asyncio.gather(consumer(async_stream=async_stream), async_stream.run(executor=None))

                    case _:
                        raise ChatModelNotImplementedException(msg="Stream type is not AsyncGenerator or Generator.")
            except ChatLengthException as e:
                api_logger.info(f"ChatLengthException: {e.msg}")
                if e.msg is None:
                    return
                buffer.current_user_chat_context.ensure_token_not_exceed(
                    extra_token_margin=buffer.current_user_chat_context.get_tokens_of(stream_progress.response)
                )
                buffer.current_user_chat_context.clear_tokens(tokens_to_remove=ChatConfig.extra_token_margin)
                await transmission()

        async def good_bye() -> None:
            await cls.message(
                websocket=buffer.websocket,
                msg=stream_progress.buffer,
                chat_room_id=buffer.current_chat_room_id,
                finish=finish,
                is_user=is_user,
                model_name=model_name,
            )

        try:
            await hand_shake()
            await transmission()

        finally:
            await good_bye()
