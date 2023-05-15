import asyncio

from app.dependencies import process_manager, process_pool_executor
from app.errors.gpt_exceptions import (
    GptInterruptedException,
    GptModelNotImplementedException,
    GptOtherException,
    GptTextGenerationException,
    GptTooMuchTokenException,
)
from app.models.gpt_llms import LlamaCppModel, LLMModel, OpenAIModel
from app.models.gpt_models import GptRoles
from app.utils.api.translate import Translator
from app.utils.chatgpt.chatgpt_buffer import BufferedUserContext
from app.utils.chatgpt.chatgpt_cache_manager import ChatGptCacheManager
from app.utils.chatgpt.chatgpt_generation import (
    generate_from_llama_cpp,
    generate_from_openai,
    message_history_organizer,
)
from app.utils.chatgpt.chatgpt_llama_cpp import llama_cpp_generation
from app.utils.chatgpt.chatgpt_message_manager import MessageManager
from app.utils.chatgpt.chatgpt_websocket_manager import SendToWebsocket


class MessageHandler:
    @staticmethod
    async def user(
        msg: str,
        translate: bool,
        buffer: BufferedUserContext,
    ) -> None:
        """Handle user message, including translation"""
        if len(buffer.current_user_gpt_context.user_message_histories) == 0:
            buffer.current_user_gpt_context.user_gpt_profile.chat_room_name = msg[:20]
            await ChatGptCacheManager.update_profile(user_gpt_context=buffer.current_user_gpt_context)
            await SendToWebsocket.init(
                buffer=buffer,
                send_previous_chats=False,
                init_callback=False,
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
                model_name=buffer.current_user_gpt_context.gpt_model.value.name,
            )
        user_token: int = buffer.current_user_gpt_context.get_tokens_of(msg)
        if user_token > buffer.current_user_gpt_context.token_per_request:  # if user message is too long
            raise GptTooMuchTokenException(
                msg=f"Message too long. Now {user_token} tokens, but {buffer.current_user_gpt_context.token_per_request} tokens allowed."
            )
        await MessageManager.add_message_history_safely(
            user_gpt_context=buffer.current_user_gpt_context,
            content=msg,
            role=GptRoles.USER,
        )

    @staticmethod
    async def gpt(
        translate: bool,
        buffer: BufferedUserContext,
    ) -> None:
        """Handle gpt message, including text generation and translation"""
        current_model: LLMModel = buffer.current_user_gpt_context.gpt_model.value
        try:
            if isinstance(current_model, OpenAIModel):
                msg: str = await SendToWebsocket.stream(
                    buffer=buffer,
                    stream=generate_from_openai(user_gpt_context=buffer.current_user_gpt_context),
                    finish=False if translate else True,
                    model_name=current_model.name,
                )
            elif isinstance(buffer.current_user_gpt_context.gpt_model.value, LlamaCppModel):
                m_queue, m_done = process_manager.Queue(), process_manager.Event()
                # async_queue, async_done = asyncio.Queue(), asyncio.Event()
                loop = asyncio.get_event_loop()
                prompt: str = message_history_organizer(
                    user_gpt_context=buffer.current_user_gpt_context,
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
                                user_gpt_context=buffer.current_user_gpt_context,
                                m_queue=m_queue,
                                m_done=m_done,
                            ),
                        ),
                        loop.run_in_executor(
                            process_pool_executor,
                            llama_cpp_generation,
                            buffer.current_user_gpt_context.gpt_model.value,
                            prompt,
                            m_queue,
                            m_done,
                            buffer.current_user_gpt_context,
                        ),
                    )
                except Exception as e:
                    raise e
                finally:
                    m_done.set()

            else:
                raise GptModelNotImplementedException(msg="Model not implemented. Please contact administrator.")
        except InterruptedError as e:
            raise GptInterruptedException(msg=str(e))
        except Exception:
            raise GptTextGenerationException(msg="An error occurred while generating text.")
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
                        model_name=buffer.current_user_gpt_context.gpt_model.value.name,
                    )
            except Exception:
                raise GptOtherException(msg="번역하는데 문제가 발생했습니다.")
