import asyncio
from copy import deepcopy

from app.dependencies import process_manager, process_pool_executor
from app.errors.chat_exceptions import (
    ChatException,
    ChatInterruptedException,
    ChatModelNotImplementedException,
    ChatOtherException,
    ChatTextGenerationException,
    ChatTooMuchTokenException,
)
from app.models.llms import LlamaCppModel, LLMModel, OpenAIModel
from app.models.chat_models import ChatRoles, UserChatContext
from app.utils.api.translate import Translator
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.cache_manager import CacheManager
from app.utils.chat.text_generation import (
    generate_from_llama_cpp,
    generate_from_openai,
    message_history_organizer,
)
from app.utils.chat.llama_cpp import llama_cpp_generation
from app.utils.chat.message_manager import MessageManager
from app.utils.chat.websocket_manager import SendToWebsocket
from app.utils.logger import api_logger


class MessageHandler:
    @staticmethod
    async def user(
        msg: str,
        translate: bool,
        buffer: BufferedUserContext,
    ) -> None:
        """Handle user message, including translation"""
        if len(buffer.current_user_chat_context.user_message_histories) == 0:
            buffer.current_user_chat_context.user_chat_profile.chat_room_name = msg[:20]
            await CacheManager.update_profile(user_chat_context=buffer.current_user_chat_context)
            await SendToWebsocket.init(
                buffer=buffer,
                send_chat_rooms=True,
                wait_next_query=True,
            )
        if translate:  # if user message is translated
            msg = await Translator.auto_translate(
                text=msg,
                src_lang="ko",
                trg_lang="en",
            )
            await SendToWebsocket.message(
                websocket=buffer.websocket,
                msg=f"## 번역된 질문\n\n{msg}\n\n## 생성된 답변\n\n",
                chat_room_id=buffer.current_chat_room_id,
                finish=False,
                model_name=buffer.current_user_chat_context.llm_model.value.name,
            )
        user_token: int = buffer.current_user_chat_context.get_tokens_of(msg)
        if user_token > buffer.current_user_chat_context.token_per_request:  # if user message is too long
            raise ChatTooMuchTokenException(
                msg=f"Message too long. Now {user_token} tokens, but {buffer.current_user_chat_context.token_per_request} tokens allowed."
            )
        await MessageManager.add_message_history_safely(
            user_chat_context=buffer.current_user_chat_context,
            content=msg,
            role=ChatRoles.USER,
        )

    @staticmethod
    async def ai(
        translate: bool,
        buffer: BufferedUserContext,
    ) -> None:
        """Handle ai message, including text generation and translation"""
        current_model: LLMModel = buffer.current_user_chat_context.llm_model.value
        backup_context: UserChatContext = deepcopy(buffer.current_user_chat_context)
        try:
            if isinstance(current_model, OpenAIModel):
                msg: str = await SendToWebsocket.stream(
                    buffer=buffer,
                    stream=generate_from_openai(user_chat_context=buffer.current_user_chat_context),
                    finish=False if translate else True,
                    model_name=current_model.name,
                )
            elif isinstance(buffer.current_user_chat_context.llm_model.value, LlamaCppModel):
                m_queue, m_done = process_manager.Queue(), process_manager.Event()
                # async_queue, async_done = asyncio.Queue(), asyncio.Event()
                loop = asyncio.get_event_loop()
                prompt: str = message_history_organizer(
                    user_chat_context=buffer.current_user_chat_context,
                    return_as_string=True,
                )  # type: ignore
                try:
                    msg, _ = await asyncio.gather(
                        SendToWebsocket.stream(
                            buffer=buffer,
                            finish=False if translate else True,
                            chunk_size=1,
                            model_name=current_model.name,
                            stream=generate_from_llama_cpp(
                                user_chat_context=buffer.current_user_chat_context,
                                m_queue=m_queue,
                                m_done=m_done,
                            ),
                        ),
                        loop.run_in_executor(
                            process_pool_executor,
                            llama_cpp_generation,
                            buffer.current_user_chat_context.llm_model.value,
                            prompt,
                            m_queue,
                            m_done,
                            buffer.current_user_chat_context,
                        ),
                    )
                except Exception as e:
                    raise e
                finally:
                    m_done.set()

            else:
                raise ChatModelNotImplementedException(msg="Model not implemented. Please contact administrator.")
        except ChatException as chat_exception:
            buffer.current_user_chat_context.copy_from(backup_context)
            raise ChatTextGenerationException(msg=chat_exception.msg)
        except InterruptedError as interrupted_error:
            buffer.current_user_chat_context.copy_from(backup_context)
            raise ChatInterruptedException(msg=str(interrupted_error))
        except Exception as exception:
            api_logger.error(f"unexpected chat exception: {exception}")
            buffer.current_user_chat_context.copy_from(backup_context)
            raise ChatTextGenerationException()
        else:
            try:
                if translate:  # if user message is translated
                    translated_msg = await Translator.auto_translate(
                        text=msg,
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
        finally:
            if "m_queue" in locals():
                del m_queue  # type: ignore
            if "m_done" in locals():
                del m_done  # type: ignore
