from typing import Optional

from app.common.config import ChatConfig
from app.shared import Shared
from app.utils.logger import ApiLogger


def get_summarization(
    to_summarize: str,
    to_summarize_tokens: Optional[int] = None,
) -> str:
    shared = Shared()
    if to_summarize_tokens is None:
        to_summarize_tokens = len(
            shared.token_text_splitter._tokenizer.encode(to_summarize)
        )

    if to_summarize_tokens < ChatConfig.summarization_token_limit:
        summarize_chain = shared.stuff_summarize_chain
    else:
        summarize_chain = shared.map_reduce_summarize_chain
    result: str = summarize_chain.run(
        shared.token_text_splitter.create_documents(
            [to_summarize],
            tokens_per_chunk=ChatConfig.summarization_chunk_size,
            chunk_overlap=ChatConfig.summarization_token_overlap,
        )
    )
    ApiLogger("||get_summarization||").info(result)
    return result
