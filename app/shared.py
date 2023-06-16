from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Manager
from multiprocessing.managers import SyncManager

from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain, stuff_prompt

from langchain.utilities import SearxSearchWrapper

from app.common.config import OPENAI_API_KEY, ChatConfig, SingletonMetaClass, config
from app.common.constants import SummarizationTemplates
from app.models.openai_functions import OpenAIFunctions
from app.utils.langchain.web_search import DuckDuckGoSearchAPIWrapper
from app.utils.langchain.token_text_splitter import CustomTokenTextSplitter
from app.utils.langchain.chat_openai import CustomChatOpenAI


@dataclass
class Shared(metaclass=SingletonMetaClass):
    process_manager: SyncManager = field(default_factory=Manager)
    process_pool_executor: ProcessPoolExecutor = field(
        default_factory=ProcessPoolExecutor
    )
    llm: CustomChatOpenAI = field(init=False)
    browsing_llm: CustomChatOpenAI = field(init=False)
    web_search_llm: CustomChatOpenAI = field(init=False)
    vectorstore_search_llm: CustomChatOpenAI = field(init=False)
    control_web_page_llm: CustomChatOpenAI = field(init=False)
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
        common_llm_kwargs = {
            "model_name": ChatConfig.global_openai_model,  # type: ignore
            "openai_api_key": OPENAI_API_KEY,
            "streaming": False,
        }
        self.llm = CustomChatOpenAI(**common_llm_kwargs)
        self.browsing_llm = CustomChatOpenAI(
            model_kwargs={
                "functions": [OpenAIFunctions.WEB_BROWSING],
                "function_call": OpenAIFunctions.WEB_BROWSING,
            },
            **common_llm_kwargs,
        )
        self.web_search_llm = CustomChatOpenAI(
            model_kwargs={
                "functions": [OpenAIFunctions.WEB_SEARCH],
                "function_call": OpenAIFunctions.WEB_SEARCH,
            },
            **common_llm_kwargs,
        )
        self.vectorstore_search_llm = CustomChatOpenAI(
            model_kwargs={
                "functions": [OpenAIFunctions.VECTORSTORE_SEARCH],
                "function_call": OpenAIFunctions.VECTORSTORE_SEARCH,
            },
            **common_llm_kwargs,
        )
        self.control_web_page_llm = CustomChatOpenAI(
            model_kwargs={
                "functions": [OpenAIFunctions.CONTROL_WEB_PAGE],
                "function_call": OpenAIFunctions.CONTROL_WEB_PAGE,
            },
            **common_llm_kwargs,
        )
        self.map_reduce_summarize_chain = load_summarize_chain(  # type: ignore
            self.llm,
            chain_type="map_reduce",
            map_prompt=stuff_prompt.PROMPT,
            combine_prompt=SummarizationTemplates.TEXT__MARKUP,
            verbose=config.debug,
        )
        self.stuff_summarize_chain = load_summarize_chain(  # type: ignore
            self.llm,
            chain_type="stuff",
            prompt=SummarizationTemplates.TEXT__MARKUP,
            verbose=config.debug,
        )
