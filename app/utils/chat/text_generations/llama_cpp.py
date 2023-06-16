from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Generator, Optional


from app.models.chat_models import MessageHistory

from app.shared import Shared
from app.errors.chat_exceptions import (
    ChatConnectionException,
    ChatTextGenerationException,
)
from app.models.llms import LlamaCppModel
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.llama_cpp import llama_cpp_generation
from app.utils.chat.messages.converter import (
    message_histories_to_str,
)
from app.utils.logger import api_logger


def generate_from_llama_cpp(
    buffer: BufferedUserContext,
    user_message_histories: list[MessageHistory],
    ai_message_histories: list[MessageHistory],
    system_message_histories: list[MessageHistory],
    max_tokens: int,
) -> Generator:
    llama_cpp_model = buffer.current_llm_model.value
    assert isinstance(llama_cpp_model, LlamaCppModel), type(llama_cpp_model)

    shared: Shared = Shared()
    m_queue = shared.process_manager.Queue()
    m_done = shared.process_manager.Event()
    process_pool_executor: ProcessPoolExecutor = shared.process_pool_executor
    try:
        prompt = message_histories_to_str(
            user_chat_roles=buffer.current_user_chat_roles,
            user_message_histories=user_message_histories,
            ai_message_histories=ai_message_histories,
            system_message_histories=system_message_histories,
            prefix_prompt=llama_cpp_model.prefix,
            prefix_prompt_tokens=llama_cpp_model.prefix_tokens,
            suffix_prompt=llama_cpp_model.suffix,
            suffix_prompt_tokens=llama_cpp_model.suffix_tokens,
            chat_turn_prompt=llama_cpp_model.chat_turn_prompt,
        )
        api_logger.info(f"Sending this prompt to llama_cpp:\n{prompt}")
        future_exception: Optional[BaseException] = None
        future: Future[None] = process_pool_executor.submit(
            llama_cpp_generation,
            user_chat_context=buffer.current_user_chat_context,
            prompt=prompt,
            max_tokens=max_tokens,
            m_queue=m_queue,
            m_done=m_done,
        )
        while not future.done():
            if buffer.done.is_set():
                m_done.set()
                # wait for the future to complete for a certain amount of time
                _, not_done = wait([future], return_when=FIRST_COMPLETED)

                # if future is still not done, try cancelling
                if future in not_done:
                    future.cancel()
                raise InterruptedError("Chat was interrupted by the user.")
            generation: Any = m_queue.get()
            if isinstance(generation, str):
                yield generation
            elif isinstance(generation, BaseException):
                api_logger.exception(
                    f"An error occurred during llama_cpp_generation: {generation} {type(generation)}"
                )
                future_exception = generation
            else:
                break
        if future_exception is not None:
            raise future_exception
    except BrokenProcessPool as e:
        process_pool_executor.shutdown(wait=False)
        shared.process_pool_executor = ProcessPoolExecutor()
        api_logger.exception(f"BrokenProcessPool: {e}")
        raise ChatConnectionException(msg="BrokenProcessPool")
