import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import WebSocket
from orjson import JSONDecodeError
from orjson import loads as orjson_loads
from pydantic import ValidationError

from app.database.schemas.auth import Users
from app.errors.chat_exceptions import (
    ChatException,
    ChatInterruptedException,
    ChatTextGenerationException,
)
from app.models.base_models import MessageFromWebsocket, SummarizedResult
from app.models.chat_commands import ChatCommands
from app.models.chat_models import ChatRoles, UserChatContext, UserChatProfile
from app.models.llms import LLMModels
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.chat_rooms import create_new_chat_room
from app.utils.chat.messages.handler import (
    MessageHandler,
    _interruption_event_watcher,
)
from app.utils.logger import ApiLogger

from .cache import CacheManager
from .message import MessageManager
from .vectorstore import VectorStoreManager
from .websocket import SendToWebsocket


async def _initialize_callback(user_id: str) -> list[UserChatProfile]:
    user_chat_profiles: list[
        UserChatProfile
    ] = await CacheManager.fetch_chat_profiles(user_id=user_id)
    if not user_chat_profiles:
        # create new chatroom
        return [
            (await create_new_chat_room(user_id=user_id)).user_chat_profile
        ]
    else:
        # get latest chatroom
        user_chat_profiles.sort(
            key=lambda profile: profile.created_at, reverse=True
        )
        return user_chat_profiles


async def _harvest_done_tasks(buffer: BufferedUserContext) -> None:
    """
    This function checks and handles the done tasks in buffer.task_list.
    :param buffer: BufferedUserContext object
    :return: None
    """
    harvested_tasks = set(
        task
        for task in buffer.task_list
        if task.done() and not task.cancelled()
    )
    update_tasks = []

    for task in harvested_tasks:
        try:
            task_result = task.result()
            if isinstance(task_result, SummarizedResult):
                context_index = buffer.find_index_of_chatroom(
                    task_result.chat_room_id
                )
                if context_index is None:
                    continue
                role = ChatRoles.get_static_member(task_result.role)
                context = await buffer._sorted_ctxts.at(context_index)
                message_history_index = (
                    await buffer.find_index_of_message_history(
                        user_chat_context=context,
                        role=role,
                        message_history_uuid=task_result.uuid,
                    )
                )
                if message_history_index is None:
                    continue

                update_tasks.append(
                    MessageManager.set_message_history_safely(
                        user_chat_context=context,
                        summarized_content=task_result.content,
                        index=message_history_index,
                        role=role,
                    )
                )
        except Exception as e:
            ApiLogger.cexception(
                f"Some error occurred while harvesting done tasks: {e}"
            )

    if update_tasks:
        ApiLogger.cinfo(f"Running update tasks: {update_tasks}")

    try:
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                ApiLogger.cexception(
                    f"Some error occurred while running update tasks: {result}"
                )
    except Exception as e:
        ApiLogger.cexception(
            f"Unexpected error occurred while running update tasks: {e}"
        )
    finally:
        buffer.task_list = [
            task for task in buffer.task_list if task not in harvested_tasks
        ]


@asynccontextmanager
async def _chat_cycle_context_manager(buffer: BufferedUserContext):
    try:
        # Send Start-of-Chat message to client
        await asyncio.gather(
            SendToWebsocket.init(buffer=buffer, send_tokens=True),
            _harvest_done_tasks(buffer=buffer),
        )

        # Run a chat cycle
        yield
    except ChatException as chat_exception:
        if isinstance(chat_exception, ChatTextGenerationException):
            # Send error message to client when text generation fails
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"\n\nAn error occurred while generating text: **{chat_exception.msg}**",
                chat_room_id=buffer.current_chat_room_id,
                finish=True,
                model_name=buffer.current_user_chat_context.llm_model.value.name,
            )
        elif (
            isinstance(chat_exception, ChatInterruptedException)
            and chat_exception.msg
        ):
            # if msg is not None, save the message generated before interruption
            await MessageManager.add_message_history_safely(
                user_chat_context=buffer.current_user_chat_context,
                role=ChatRoles.AI,
                content=chat_exception.msg,
            )
        else:
            # Send error message to client when other errors occur
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=str(chat_exception.msg),
                chat_room_id=buffer.current_chat_room_id,
            )
    finally:
        # Clean up buffer
        buffer.done.clear()
        buffer.optional_info["uuid"] = None

        # Send End-of-Chat message to client
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg="",
            chat_room_id=buffer.current_chat_room_id,
            finish=True,
            model_name=buffer.current_user_chat_context.llm_model.value.name,
        )


async def _change_context(
    buffer: BufferedUserContext, changed_chat_room_id: str
) -> None:
    index: int | None = buffer.find_index_of_chatroom(changed_chat_room_id)
    if index is None:
        # if received chat_room_id is not in chat_room_ids, create new chat room
        await create_new_chat_room(
            user_id=buffer.user_id,
            new_chat_room_id=changed_chat_room_id,
            buffer=buffer,
        )
        await SendToWebsocket.init(
            buffer=buffer,
            send_chat_rooms=True,
            send_previous_chats=True,
        )
    else:
        # if received chat_room_id is in chat_room_ids, get context from memory
        await buffer.change_context_to(index=index)
        await SendToWebsocket.init(
            buffer=buffer,
            send_previous_chats=True,
        )


async def _handle_json_reception(
    dict_json: dict, buffer: BufferedUserContext
) -> None:
    if "filename" in dict_json:
        # if received json has filename, it is a file
        buffer.optional_info["filename"] = dict_json["filename"]
    elif "chat_room_name" in dict_json and "chat_room_id" in dict_json:
        chat_room_idx = buffer.find_index_of_chatroom(
            chat_room_id=dict_json["chat_room_id"]
        )
        if chat_room_idx is not None:
            user_chat_context_or_profile = buffer[chat_room_idx]
            if isinstance(user_chat_context_or_profile, UserChatContext):
                user_chat_profile = (
                    user_chat_context_or_profile.user_chat_profile
                )
            else:
                user_chat_profile = user_chat_context_or_profile
            user_chat_profile.chat_room_name = dict_json["chat_room_name"][:20]
            await CacheManager.update_profile(
                user_chat_profile=user_chat_profile
            )
            await SendToWebsocket.init(buffer=buffer, send_chat_rooms=True)
    elif "model" in dict_json:
        found_model = LLMModels.get_member(
            dict_json["model"],
        )
        buffer.current_user_chat_context.llm_model = found_model
        await SendToWebsocket.init(buffer=buffer, send_selected_model=True)
        await CacheManager.update_model(
            user_chat_context=buffer.current_user_chat_context
        )


async def _handle_text_reception(
    text: str, buffer: BufferedUserContext
) -> None:
    if text == "stop":
        buffer.done.set()


@asynccontextmanager
async def _websocket_context_manager(buffer: BufferedUserContext):
    try:
        yield
    except Exception as e:
        if isinstance(e, RuntimeError) and "receive" in str(e):
            return
        ApiLogger.cerror(f"Exception in chat: {e}", exc_info=True)
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg="Internal server error. Please try again.",
            chat_room_id=buffer.current_chat_room_id,
        )
        raise e


async def _websocket_receiver(buffer: BufferedUserContext) -> None:
    while True:  # loop until connection is closed
        received = await buffer.websocket.receive()
        received_text: Optional[str] = received.get("text")
        received_bytes: Optional[bytes] = received.get("bytes")

        if received_text is not None:
            try:
                received_json = orjson_loads(received_text)
                assert isinstance(received_json, dict)
            except (JSONDecodeError, AssertionError):
                await _handle_text_reception(text=received_text, buffer=buffer)
                continue
            try:
                await buffer.queue.put(MessageFromWebsocket(**received_json))
            except ValidationError:
                await _handle_json_reception(
                    dict_json=received_json, buffer=buffer
                )

        if received_bytes is not None:
            await buffer.queue.put(
                await VectorStoreManager.embed_file_to_vectorstore(
                    file=received_bytes,
                    filename=buffer.optional_info.get("filename") or "",
                    collection_name=buffer.current_user_chat_context.user_id,
                )
            )


async def _websocket_sender(buffer: BufferedUserContext) -> None:
    while True:  # loop until connection is closed
        async with _chat_cycle_context_manager(buffer=buffer):
            item: MessageFromWebsocket | str = await buffer.queue.get()
            await _harvest_done_tasks(buffer=buffer)

            if isinstance(item, MessageFromWebsocket):
                if item.chat_room_id != buffer.current_chat_room_id:
                    # This is a message from another chat room, interpreted as change of context, ignoring message
                    await _change_context(
                        buffer=buffer,
                        changed_chat_room_id=item.chat_room_id,
                    )
                elif item.msg.startswith("/"):
                    # if user message is command, handle command
                    buffer.optional_info["uuid"] = item.uuid
                    buffer.optional_info["translate"] = item.translate
                    splitted: list[str] = item.msg[1:].split(" ")
                    if not item.msg.startswith("/") or not any(
                        (
                            item.msg[1:].startswith(command)
                            for command in ChatCommands.special_commands
                        )
                    ):
                        buffer.last_user_message = item.msg
                    await _interruption_event_watcher(
                        MessageHandler.command(
                            callback_name=splitted[0],
                            callback_args=splitted[1:],
                            callback_finder=ChatCommands.find_callback_with_command,
                            buffer=buffer,
                        ),
                        event=buffer.done,
                    )
                else:
                    buffer.optional_info["uuid"] = item.uuid
                    buffer.optional_info["translate"] = item.translate
                    buffer.last_user_message = item.msg
                    await MessageHandler.user(msg=item.msg, buffer=buffer)
                    await MessageHandler.ai(buffer=buffer)
            else:
                await SendToWebsocket.message(
                    websocket=buffer.websocket,
                    msg=str(item),
                    chat_room_id=buffer.current_chat_room_id,
                )


class ChatStreamManager:
    @classmethod
    async def begin_chat(cls, websocket: WebSocket, user: Users) -> None:
        # initialize buffer
        buffer: BufferedUserContext = BufferedUserContext(
            user=user,
            websocket=websocket,
            initialize_callback=_initialize_callback,
            read_callback=CacheManager.read_context_from_profile,
        )
        await buffer.init()

        # initialize websocket
        await SendToWebsocket.init(
            buffer=buffer,
            send_chat_rooms=True,
            send_previous_chats=True,
            send_models=True,
            send_selected_model=True,
        )

        # start chat
        async with _websocket_context_manager(buffer=buffer):
            await asyncio.gather(
                _websocket_receiver(buffer=buffer),
                _websocket_sender(buffer=buffer),
            )
