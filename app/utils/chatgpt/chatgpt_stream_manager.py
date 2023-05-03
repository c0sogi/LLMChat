import asyncio
from uuid import uuid4
import orjson
from typing import AsyncGenerator, Callable
from inspect import iscoroutinefunction
from fastapi import WebSocket, WebSocketDisconnect
from app.errors.gpt_exceptions import (
    GptTooMuchTokenException,
    GptTextGenerationException,
    GptOtherException,
)
from app.utils.chatgpt.chatgpt_message_manager import MessageManager
from app.utils.logger import api_logger
from app.utils.tests.timeit import timeit
from app.viewmodels.base_models import MessageToWebsocket, MessageFromWebsocket
from app.viewmodels.gpt_models import GptRoles, UserGptContext
from app.utils.chatgpt.chatgpt_commands import ChatGptCommands
from app.utils.chatgpt.chatgpt_cache_manager import chatgpt_cache_manager
from app.utils.chatgpt.chatgpt_generation import generate_from_openai, message_history_organizer
from app.utils.api.translate import papago_translate_api


async def create_new_chat_room(
    user_id: str,
    client_chat_room_id: str | None = None,
    sorted_user_gpt_contexts: list[UserGptContext] | None = None,
    sorted_chat_room_ids: list[str] | None = None,
) -> UserGptContext:
    client_chat_room_id = client_chat_room_id if client_chat_room_id else uuid4().hex
    default: UserGptContext = UserGptContext.construct_default(
        user_id=user_id,
        chat_room_id=client_chat_room_id,
    )
    await chatgpt_cache_manager.create_context(default)
    if sorted_user_gpt_contexts is not None and sorted_chat_room_ids is not None:
        sorted_user_gpt_contexts.insert(0, default)
        sorted_chat_room_ids.insert(0, client_chat_room_id)
        assert len(sorted_chat_room_ids) == len(sorted_user_gpt_contexts)
    return default


async def delete_chat_room(
    user_id: str,
    chat_room_id: str,
    sorted_user_gpt_contexts: list[UserGptContext] | None = None,
    sorted_chat_room_ids: list[str] | None = None,
) -> None:
    await chatgpt_cache_manager.delete_chat_room(user_id=user_id, chat_room_id=chat_room_id)
    if sorted_user_gpt_contexts is not None and sorted_chat_room_ids is not None:
        index = sorted_chat_room_ids.index(chat_room_id)
        del sorted_user_gpt_contexts[index]
        del sorted_chat_room_ids[index]
        assert len(sorted_chat_room_ids) == len(sorted_user_gpt_contexts)


async def get_contexts_sorted_from_recent_to_past(user_id: str, chat_room_ids: list[str]) -> list[UserGptContext]:
    if len(chat_room_ids) == 0:
        # create new chatroom
        return [await create_new_chat_room(user_id=user_id)]
    else:
        # get latest chatroom
        contexts: list[UserGptContext] = await asyncio.gather(
            *[
                chatgpt_cache_manager.read_context(user_id=user_id, chat_room_id=chat_room_id)
                for chat_room_id in chat_room_ids
            ]
        )
        contexts.sort(key=lambda x: x.user_gpt_profile.created_at, reverse=True)
        return contexts


async def begin_chat(
    websocket: WebSocket,
    user_id: str,
    openai_api_key: str,
) -> None:  # websocket for chat gpt
    # initialize variables
    sorted_user_gpt_contexts: list[UserGptContext] = await get_contexts_sorted_from_recent_to_past(
        user_id=user_id, chat_room_ids=await chatgpt_cache_manager.get_all_chat_rooms(user_id=user_id)
    )
    # sort chat_room_ids by created_at
    sorted_chat_room_ids: list[str] = [context.chat_room_id for context in sorted_user_gpt_contexts]
    current_user_gpt_context: UserGptContext = sorted_user_gpt_contexts[0]
    assert len(sorted_chat_room_ids) == len(sorted_user_gpt_contexts)

    await SendToWebsocket.initiation_of_chat(
        websocket=websocket,
        user_gpt_context=current_user_gpt_context,
        sorted_chat_room_ids=sorted_chat_room_ids,
    )

    while True:  # loop until connection is closed
        try:
            # initialize variables
            received: MessageFromWebsocket = MessageFromWebsocket.parse_raw(await websocket.receive_text())
            msg: str = received.msg
            translate: bool = received.translate
            client_chat_room_id: str = received.chat_room_id

            if msg == "/deletechatroom":  # delete chat room
                await delete_chat_room(
                    user_id=user_id,
                    chat_room_id=client_chat_room_id,
                    sorted_user_gpt_contexts=sorted_user_gpt_contexts,
                    sorted_chat_room_ids=sorted_chat_room_ids,
                )

                if len(sorted_chat_room_ids) == 0:
                    await create_new_chat_room(
                        user_id=user_id,
                        sorted_user_gpt_contexts=sorted_user_gpt_contexts,
                        sorted_chat_room_ids=sorted_chat_room_ids,
                    )
                current_user_gpt_context = sorted_user_gpt_contexts[0]

                await SendToWebsocket.initiation_of_chat(
                    websocket=websocket,
                    user_gpt_context=current_user_gpt_context,
                    sorted_chat_room_ids=sorted_chat_room_ids,
                )
                continue

            if client_chat_room_id != current_user_gpt_context.chat_room_id:  # change chat room
                if client_chat_room_id not in sorted_chat_room_ids:
                    # if received chat_room_id is not in chat_room_ids, create new chat room
                    current_user_gpt_context = await create_new_chat_room(
                        user_id=user_id,
                        sorted_user_gpt_contexts=sorted_user_gpt_contexts,
                        sorted_chat_room_ids=sorted_chat_room_ids,
                    )
                else:
                    # if received chat_room_id is in chat_room_ids, get context from memory
                    current_user_gpt_context = sorted_user_gpt_contexts[
                        sorted_chat_room_ids.index(client_chat_room_id)
                    ]  # get context from sorted_user_gpt_contexts
                await SendToWebsocket.initiation_of_chat(
                    websocket=websocket,
                    user_gpt_context=current_user_gpt_context,
                    sorted_chat_room_ids=sorted_chat_room_ids,
                )

            if msg.startswith("/"):  # if user message is command
                command_response: str | None = await get_command_response(
                    msg=msg, user_gpt_context=current_user_gpt_context
                )
                if command_response is not None:
                    await SendToWebsocket.message(
                        websocket=websocket,
                        msg=command_response,
                        chat_room_id=current_user_gpt_context.chat_room_id,
                    )  # send callback response to websocket
                    continue

            else:  # handle normal message
                await HandleMessage.user(
                    websocket=websocket,
                    user_gpt_context=current_user_gpt_context,
                    msg=msg,
                    translate=translate,
                    chat_room_id=current_user_gpt_context.chat_room_id,
                )
            await HandleMessage.gpt(
                websocket=websocket,
                user_gpt_context=current_user_gpt_context,
                translate=translate,
                chat_room_id=current_user_gpt_context.chat_room_id,
                openai_api_key=openai_api_key,
            )

        except WebSocketDisconnect:
            raise WebSocketDisconnect(code=1000, reason="client disconnected")
        except GptTextGenerationException:
            await MessageManager.rpop_message_history_safely(
                user_gpt_context=current_user_gpt_context, role=GptRoles.USER
            )
        except GptOtherException:
            await MessageManager.rpop_message_history_safely(
                user_gpt_context=current_user_gpt_context, role=GptRoles.USER
            )
            await MessageManager.rpop_message_history_safely(
                user_gpt_context=current_user_gpt_context, role=GptRoles.GPT
            )
        except GptTooMuchTokenException as too_much_token_exception:  # if user message is too long
            await SendToWebsocket.message(
                websocket=websocket,
                msg=too_much_token_exception.msg if too_much_token_exception.msg is not None else "",
                chat_room_id=current_user_gpt_context.chat_room_id,
            )  # send too much token exception message to websocket
            continue
        except Exception as exception:  # if other exception is raised
            api_logger.error(f"chat exception: {exception}", exc_info=True)
            await websocket.send_json(  # finish stream message
                MessageToWebsocket(
                    msg="Internal Server Error",
                    finish=True,
                    chat_room_id=current_user_gpt_context.chat_room_id,
                    is_user=False,
                ).dict()
            )
            break


class SendToWebsocket:
    @staticmethod
    async def initiation_of_chat(
        websocket: WebSocket,
        sorted_chat_room_ids: list[str],
        user_gpt_context: UserGptContext,
    ) -> None:
        await SendToWebsocket.message(
            websocket=websocket,
            msg=orjson.dumps(
                {
                    "previous_chats": message_history_organizer(
                        user_gpt_context=user_gpt_context,
                        send_to_openai=False,
                    ),
                    "chat_room_ids": sorted_chat_room_ids,
                }
            ).decode("utf-8"),
            chat_room_id=user_gpt_context.chat_room_id,
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
    ) -> None:  # send whole message to websocket
        await websocket.send_json(  # send stream message
            MessageToWebsocket(
                msg=msg,
                finish=finish,
                chat_room_id=chat_room_id,
                is_user=is_user,
                init=init,
            ).dict()
        )

    @staticmethod
    async def stream(
        websocket: WebSocket,
        stream: AsyncGenerator,
        chat_room_id: str,
        finish: bool = True,
        is_user: bool = False,
        chunk_size: int = 3,
    ) -> str:  # send whole stream to websocket
        response: str = ""
        chunk_buffer: list[str] = []
        async for delta in stream:  # stream from api
            response += delta
            chunk_buffer.append(delta)
            if len(chunk_buffer) >= chunk_size:
                await websocket.send_json(
                    MessageToWebsocket(
                        msg="".join(chunk_buffer),
                        finish=False,
                        chat_room_id=chat_room_id,
                        is_user=is_user,
                    ).dict()
                )
                chunk_buffer.clear()
        if len(chunk_buffer) > 0:
            # flush remaining chunks
            await websocket.send_json(
                MessageToWebsocket(
                    msg="".join(chunk_buffer),
                    finish=False,
                    chat_room_id=chat_room_id,
                    is_user=is_user,
                ).dict()
            )
        if finish:
            await websocket.send_json(  # finish stream message
                MessageToWebsocket(
                    msg="",
                    finish=True,
                    chat_room_id=chat_room_id,
                    is_user=is_user,
                ).dict()
            )
        return response


class HandleMessage:
    @staticmethod
    async def user(
        websocket: WebSocket,
        user_gpt_context: UserGptContext,
        msg: str,
        translate: bool,
        chat_room_id: str,
    ) -> None:
        if translate:  # if user message is translated
            msg = await papago_translate_api(
                text=msg,
                src_lang="ko",
                trg_lang="en",
            )
            await SendToWebsocket.message(
                websocket=websocket,
                msg=f"[번역된 질문]\n\n{msg}",
                chat_room_id=chat_room_id,
            )
        user_token: int = len(user_gpt_context.tokenize(msg))
        if user_token > user_gpt_context.token_per_request:  # if user message is too long
            raise GptTooMuchTokenException(
                msg=f"메시지가 너무 길어요. 현재 토큰 개수는 {user_token}로, {user_gpt_context.token_per_request} 이하여야 합니다."
            )
        await MessageManager.add_message_history_safely(
            user_gpt_context=user_gpt_context,
            content=msg,
            role=GptRoles.USER,
        )

    @staticmethod
    async def gpt(
        websocket: WebSocket,
        user_gpt_context: UserGptContext,
        translate: bool,
        chat_room_id: str,
        openai_api_key: str,
    ) -> None:
        try:
            msg: str = await SendToWebsocket.stream(
                websocket=websocket,
                chat_room_id=chat_room_id,
                stream=generate_from_openai(
                    user_gpt_context=user_gpt_context,
                    openai_api_key=openai_api_key,
                ),
            )
        except Exception:
            raise GptTextGenerationException(msg="텍스트를 생성하는데 문제가 발생했습니다.")
        try:
            if translate:  # if user message is translated
                translated_msg = await papago_translate_api(
                    text=msg,
                    src_lang="en",
                    trg_lang="ko",
                )
                await SendToWebsocket.message(
                    websocket=websocket,
                    msg=f"[번역된 답변]\n\n{translated_msg}",
                    chat_room_id=chat_room_id,
                )
        except Exception:
            raise GptOtherException(msg="번역하는데 문제가 발생했습니다.")


async def get_command_response(msg: str, user_gpt_context: UserGptContext) -> str | None:
    user_command: list = msg.split(" ")
    callback: Callable = (
        getattr(ChatGptCommands, user_command[0][1:])
        if hasattr(ChatGptCommands, user_command[0][1:])
        else ChatGptCommands.not_existing_callback
    )  # get callback function
    callback_args: list = user_command[1:]  # command args
    if iscoroutinefunction(callback):  # if callback is coroutine function
        callback_response: str = await callback(*callback_args, user_gpt_context=user_gpt_context)
    else:
        callback_response: str = callback(*callback_args, user_gpt_context=user_gpt_context)
    return callback_response
