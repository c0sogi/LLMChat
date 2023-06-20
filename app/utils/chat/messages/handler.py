import asyncio
from copy import deepcopy
from typing import Optional
from uuid import uuid4

from fastapi.concurrency import run_in_threadpool

from app.common.config import ChatConfig, config
from app.errors.chat_exceptions import (
    ChatException,
    ChatStreamingInterruptedException,
    ChatModelNotImplementedException,
    ChatTextGenerationException,
    ChatTooMuchTokenException,
)
from app.models.chat_models import ChatRoles, UserChatContext
from app.models.llms import LlamaCppModel, LLMModel, OpenAIModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.cache import CacheManager
from app.utils.chat.chains.translate import translate_chain
from app.utils.chat.managers.message import MessageManager
from app.utils.chat.text_generations.completion_api import (
    agenerate_from_chat_completion_api,
    agenerate_from_text_completion_api,
)
from app.utils.chat.text_generations.summarization import get_summarization
from app.utils.chat.text_generations.openai import agenerate_from_openai
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.date_utils import UTC
from app.utils.logger import api_logger
from app.models.base_models import StreamProgress, SummarizedResult


async def summarization_task(
    user_id: str,
    chat_room_id: str,
    role: str,
    to_summarize: str,
    message_history_uuid: str,
) -> SummarizedResult:  # =
    return SummarizedResult(
        user_id=user_id,
        chat_room_id=chat_room_id,
        role=role,
        content=await run_in_threadpool(get_summarization, to_summarize=to_summarize),
        uuid=message_history_uuid,
    )


class MessageHandler:
    @staticmethod
    async def user(
        msg: str,
        translate: Optional[str],
        buffer: BufferedUserContext,
        use_tight_token_limit: bool = True,
    ) -> None:
        """Handle user message, including translation"""
        if not buffer.current_user_message_histories and UTC.check_string_valid(
            buffer.current_chat_room_name
        ):
            buffer.current_chat_room_name = msg[:20]
            await CacheManager.update_profile(
                user_chat_context=buffer.current_user_chat_context
            )
            await SendToWebsocket.init(
                buffer=buffer, send_chat_rooms=True, wait_next_query=True
            )

        if translate:  # if user message is translated
            translate_chain_result: Optional[str] = await translate_chain(
                buffer=buffer,
                query=msg,
                finish=False,
                wait_next_query=False,
                show_result=True,
                src_lang=translate,
                trg_lang="en",
            )
            if translate_chain_result is not None:
                msg = translate_chain_result

        user_token: int = buffer.current_user_chat_context.get_tokens_of(msg)
        current_llm_model: LLMModel = buffer.current_llm_model.value
        token_limit: int = (
            current_llm_model.max_tokens_per_request
            if use_tight_token_limit
            else current_llm_model.max_total_tokens - ChatConfig.extra_token_margin
        )
        if user_token > token_limit:  # if user message is too long
            raise ChatTooMuchTokenException(
                msg=f"Message too long. Now {user_token} tokens, "
                f"but {token_limit} tokens allowed."
            )
        await MessageManager.add_message_history_safely(
            user_chat_context=buffer.current_user_chat_context,
            content=msg,
            role=ChatRoles.USER,
            calculated_tokens_to_use=user_token + current_llm_model.token_margin,
        )
        # asyncio.create_task(asyncio.sleep(10))
        if (
            ChatConfig.summarize_for_chat
            and buffer.current_user_message_histories[-1].tokens
            > ChatConfig.summarization_threshold
        ):
            buffer.task_list.append(
                asyncio.create_task(
                    summarization_task(
                        user_id=buffer.user_id,
                        chat_room_id=buffer.current_chat_room_id,
                        role="user",
                        to_summarize=buffer.current_user_message_histories[-1].content,
                        message_history_uuid=buffer.current_user_message_histories[
                            -1
                        ].uuid,
                    )
                )
            )  # =

    @classmethod
    async def ai(cls, translate: Optional[str], buffer: BufferedUserContext) -> None:
        """Handle ai message, including text generation and translation"""
        backup_context: UserChatContext = deepcopy(buffer.current_user_chat_context)
        current_model = buffer.current_user_chat_context.llm_model.value
        stream_progress = StreamProgress(uuid=uuid4().hex)
        try:
            if isinstance(current_model, OpenAIModel):
                stream_func = agenerate_from_openai

            elif isinstance(current_model, LlamaCppModel):
                if config.is_llama_cpp_available and config.llama_cpp_completion_url:
                    # Use llama_cpp API
                    if "/v1/chat/completions" in config.llama_cpp_completion_url:
                        stream_func = agenerate_from_chat_completion_api
                    elif "/v1/completions" in config.llama_cpp_completion_url:
                        stream_func = agenerate_from_text_completion_api
                    else:
                        raise ChatModelNotImplementedException(
                            msg=f"Model {buffer.current_user_chat_context.llm_model.value.name} not implemented."
                        )
                else:
                    raise ChatModelNotImplementedException(
                        msg=f"Model {buffer.current_user_chat_context.llm_model.value.name} not implemented."
                    )
                    # Use llama_cpp process pool directly
                    # try:
                    #     from app.utils.chat.text_generations.llama_cpp import (
                    #         generate_from_llama_cpp,
                    #     )

                    #     stream_func = generate_from_llama_cpp
                    # except ImportError:
                    #     raise ChatModelNotImplementedException(
                    #         msg=f"Model {buffer.current_user_chat_context.llm_model.value.name} not implemented."
                    #     )
            else:
                raise ChatModelNotImplementedException(
                    msg=f"Model {buffer.current_user_chat_context.llm_model.value.name} not implemented."
                )
            await SendToWebsocket.stream(
                buffer=buffer,
                stream_func=stream_func,
                stream_progress=stream_progress,
                model_name=current_model.name,
                wait_next_query=True if translate else None,
            )
            await MessageManager.add_message_history_safely(
                user_chat_context=buffer.current_user_chat_context,
                content=stream_progress.response,
                role=ChatRoles.AI,
                uuid=stream_progress.uuid,
            )
            if (
                ChatConfig.summarize_for_chat
                and buffer.current_ai_message_histories[-1].tokens
                > ChatConfig.summarization_threshold
            ):
                buffer.task_list.append(
                    asyncio.create_task(
                        summarization_task(
                            user_id=buffer.user_id,
                            chat_room_id=buffer.current_chat_room_id,
                            role="ai",
                            to_summarize=buffer.current_ai_message_histories[
                                -1
                            ].content,
                            message_history_uuid=buffer.current_ai_message_histories[
                                -1
                            ].uuid,
                        )
                    )
                )  # =

        except ChatException as chat_exception:
            buffer.current_user_chat_context.copy_from(backup_context)
            raise ChatTextGenerationException(msg=chat_exception.msg)

        except InterruptedError as interrupted_error:
            buffer.current_user_chat_context.copy_from(backup_context)
            buffer.done.clear()
            raise ChatStreamingInterruptedException(msg=str(interrupted_error))

        except Exception as exception:
            api_logger.error(f"unexpected chat exception: {exception}", exc_info=True)
            buffer.current_user_chat_context.copy_from(backup_context)
            raise ChatTextGenerationException()

        else:
            if translate:
                await translate_chain(
                    buffer=buffer,
                    query=stream_progress.response,
                    finish=True,
                    wait_next_query=False,
                    show_result=True,
                    src_lang="en",
                    trg_lang=translate,
                )
