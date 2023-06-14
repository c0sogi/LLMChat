from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Manager
from multiprocessing.managers import SyncManager

import openai
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain, stuff_prompt
from langchain.embeddings import OpenAIEmbeddings
from langchain.utilities import SearxSearchWrapper

from app.common.config import OPENAI_API_KEY, ChatConfig, SingletonMetaClass, config
from app.common.constants import SummarizationTemplates
from app.utils.langchain.web_search import DuckDuckGoSearchAPIWrapper
from app.utils.langchain.token_text_splitter import CustomTokenTextSplitter
from app.utils.langchain.chat_openai import CustomChatOpenAI


@dataclass
class Shared(metaclass=SingletonMetaClass):
    process_manager: SyncManager = field(default_factory=Manager)
    process_pool_executor: ProcessPoolExecutor = field(
        default_factory=ProcessPoolExecutor
    )
    openai_embeddings: OpenAIEmbeddings = field(init=False)
    openai_llm: CustomChatOpenAI = field(init=False)
    map_reduce_summarize_chain: MapReduceDocumentsChain = field(init=False)
    stuff_summarize_chain: StuffDocumentsChain = field(init=False)
    token_text_splitter: CustomTokenTextSplitter = field(
        default_factory=lambda: CustomTokenTextSplitter(encoding_name="cl100k_base")
    )
    searx: SearxSearchWrapper = field(
        default_factory=lambda: SearxSearchWrapper(searx_host="http://localhost:8080")
    )
    duckduckgo: DuckDuckGoSearchAPIWrapper = field(
        default_factory=lambda: DuckDuckGoSearchAPIWrapper()
    )

    def __post_init__(self):
        self.openai_embeddings = OpenAIEmbeddings(
            client=openai.Embedding,
            openai_api_key=OPENAI_API_KEY,
        )
        self.openai_llm = CustomChatOpenAI(
            client=None,
            model_name=ChatConfig.summarization_openai_model,
            openai_api_key=OPENAI_API_KEY,
        )
        self.map_reduce_summarize_chain = load_summarize_chain(  # type: ignore
            self.openai_llm,
            chain_type="map_reduce",
            map_prompt=stuff_prompt.PROMPT,
            combine_prompt=SummarizationTemplates.TEXT__MARKUP,
            verbose=config.debug,
        )
        self.stuff_summarize_chain = load_summarize_chain(  # type: ignore
            self.openai_llm,
            chain_type="stuff",
            prompt=SummarizationTemplates.TEXT__MARKUP,
            verbose=config.debug,
        )
