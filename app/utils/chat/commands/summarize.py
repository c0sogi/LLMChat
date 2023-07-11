from time import time
from typing import Optional

from fastapi.concurrency import run_in_threadpool

from app.models.chat_models import command_response
from app.shared import Shared
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.messages.converter import message_histories_to_str
from app.utils.chat.text_generations.summarization import get_summarization


class SummarizeCommands:
    @staticmethod
    @command_response.send_message_and_stop
    async def summarize(
        to_summarize: Optional[str], /, buffer: BufferedUserContext
    ) -> str:
        shared = Shared()
        if to_summarize is None:
            to_summarize = message_histories_to_str(
                user_chat_roles=buffer.current_user_chat_roles,
                user_message_histories=buffer.current_user_message_histories,
                ai_message_histories=buffer.current_ai_message_histories,
                system_message_histories=buffer.current_system_message_histories,
            )
        to_summarize_tokens = buffer.current_user_chat_context.total_tokens
        start: float = time()
        summarized = await run_in_threadpool(
            get_summarization,
            to_summarize=to_summarize,
            to_summarize_tokens=to_summarize_tokens,
        )
        original_tokens: int = len(
            shared.token_text_splitter._tokenizer.encode(to_summarize)
        )
        summarized_tokens: int = len(
            shared.token_text_splitter._tokenizer.encode(summarized)
        )
        end: float = time()
        return "\n".join(
            (
                "# Summarization complete!",
                f"- original tokens: {original_tokens}",
                f"- summarized tokens: {summarized_tokens}",
                f"- summarization time: {end-start}s",
                "```",
                summarized,
                "```",
            )
        )
