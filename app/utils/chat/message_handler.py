import asyncio
from copy import deepcopy
from uuid import uuid4
from app.common.config import ChatConfig

from app.errors.chat_exceptions import (
    ChatException,
    ChatInterruptedException,
    ChatModelNotImplementedException,
    ChatOtherException,
    ChatTextGenerationException,
    ChatTooMuchTokenException,
)
from app.models.chat_models import ChatRoles, UserChatContext
from app.models.llms import LLMModel, LlamaCppModel, OpenAIModel
from app.utils.api.translate import Translator
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.cache_manager import CacheManager
from app.utils.chat.message_manager import MessageManager
from app.utils.chat.text_generation import (
    agenerate_from_openai,
    generate_from_llama_cpp,
    get_summarization,
)
from app.utils.chat.websocket_manager import SendToWebsocket
from app.utils.date_utils import UTC
from app.utils.logger import api_logger
from app.viewmodels.base_models import StreamProgress, SummarizedResult


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
        content=await get_summarization(to_summarize),
        uuid=message_history_uuid,
    )


class MessageHandler:
    @staticmethod
    async def user(
        msg: str,
        translate: bool,
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
            msg = await Translator.translate(text=msg, src_lang="ko", trg_lang="en")
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"## 번역된 질문\n\n{msg}\n\n## 생성된 답변\n\n",
                chat_room_id=buffer.current_chat_room_id,
                finish=False,
                model_name=buffer.current_user_chat_context.llm_model.value.name,
            )

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
        )
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
    async def ai(cls, translate: bool, buffer: BufferedUserContext) -> None:
        """Handle ai message, including text generation and translation"""
        backup_context: UserChatContext = deepcopy(buffer.current_user_chat_context)
        current_model = buffer.current_user_chat_context.llm_model.value
        stream_progress = StreamProgress(uuid=uuid4().hex)
        try:
            if isinstance(current_model, OpenAIModel):
                await SendToWebsocket.stream(
                    buffer=buffer,
                    stream_func=agenerate_from_openai,
                    stream_progress=stream_progress,
                    finish=False if translate else True,
                    model_name=current_model.name,
                )

            elif isinstance(current_model, LlamaCppModel):
                await SendToWebsocket.stream(
                    buffer=buffer,
                    stream_func=generate_from_llama_cpp,
                    stream_progress=stream_progress,
                    finish=False if translate else True,
                    model_name=current_model.name,
                )
            else:
                raise ChatModelNotImplementedException(
                    msg=f"Model {buffer.current_user_chat_context.llm_model.value.name} not implemented."
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
            raise ChatInterruptedException(msg=str(interrupted_error))

        except Exception as exception:
            api_logger.error(f"unexpected chat exception: {exception}", exc_info=True)
            buffer.current_user_chat_context.copy_from(backup_context)
            raise ChatTextGenerationException()

        else:
            try:
                if translate:  # if user message is translated
                    translated_msg = await Translator.translate(
                        text=stream_progress.response,
                        src_lang="en",
                        trg_lang="ko",
                    )
                    await SendToWebsocket.message(
                        websocket=buffer.websocket,
                        msg=f"\n\n## 번역된 답변\n\n{translated_msg}",
                        chat_room_id=buffer.current_chat_room_id,
                        model_name=buffer.current_user_chat_context.llm_model.value.name,
                    )
            except Exception:
                raise ChatOtherException(msg="번역하는데 문제가 발생했습니다.")
