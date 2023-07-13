from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Event as ProcessEvent
from multiprocessing import Manager, Process
from multiprocessing.managers import SyncManager
from multiprocessing.synchronize import Event as ProcessEventClass
from threading import Event as ThreadEvent
from threading import Thread
from typing import Optional

from langchain.chains.combine_documents.map_reduce import (
    MapReduceDocumentsChain,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain, stuff_prompt
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.utilities import SearxSearchWrapper

from app.common.config import (
    OPENAI_API_KEY,
    ChatConfig,
    SingletonMetaClass,
    config,
)
from app.common.constants import SummarizationTemplates
from app.utils.langchain.embeddings_api import APIEmbeddings
from app.utils.langchain.token_text_splitter import CustomTokenTextSplitter
from app.utils.langchain.web_search import DuckDuckGoSearchAPIWrapper


@dataclass
class Shared(metaclass=SingletonMetaClass):
    openai_embeddings: OpenAIEmbeddings = field(init=False)
    local_embeddings: Optional[APIEmbeddings] = field(init=False)
    map_reduce_summarize_chain: MapReduceDocumentsChain = field(init=False)
    stuff_summarize_chain: StuffDocumentsChain = field(init=False)
    token_text_splitter: CustomTokenTextSplitter = field(
        default_factory=lambda: CustomTokenTextSplitter(
            encoding_name="cl100k_base"
        )
    )
    searx: SearxSearchWrapper = field(
        default_factory=lambda: SearxSearchWrapper(
            searx_host="http://localhost:8080"
        )
    )
    duckduckgo: DuckDuckGoSearchAPIWrapper = field(
        default_factory=lambda: DuckDuckGoSearchAPIWrapper()
    )

    def __post_init__(self):
        self.openai_embeddings = OpenAIEmbeddings(
            client=None,
            openai_api_key=OPENAI_API_KEY,
        )
        if config.llama_embedding_url and ChatConfig.local_embedding_model:
            self.local_embeddings = APIEmbeddings(
                client=None,
                model=ChatConfig.local_embedding_model,
                embedding_api_url=config.llama_embedding_url,
            )
        else:
            self.local_embeddings = None
        self._process_manager = None
        self._process_pool_executor = None
        self._process = None
        self._process_terminate_signal = None
        self._thread = None
        self._thread_terminate_signal = None
        self.llm = ChatOpenAI(
            client=None,
            model=ChatConfig.global_openai_model,
            openai_api_key=OPENAI_API_KEY,
            streaming=False,
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

    @property
    def process_manager(self) -> SyncManager:
        if not self._process_manager:
            self._process_manager = Manager()
        return self._process_manager

    @property
    def process_pool_executor(self) -> ProcessPoolExecutor:
        if not self._process_pool_executor:
            self._process_pool_executor = ProcessPoolExecutor()
        return self._process_pool_executor

    @process_pool_executor.setter
    def process_pool_executor(self, value: ProcessPoolExecutor) -> None:
        self._process_pool_executor = value

    @property
    def process(self) -> Process:
        if not self._process:
            self._process = Process()
        return self._process

    @process.setter
    def process(self, value: Process) -> None:
        self._process = value

    @property
    def thread(self) -> Thread:
        if not self._thread:
            self._thread = Thread()
        return self._thread

    @thread.setter
    def thread(self, value: Thread) -> None:
        self._thread = value

    @property
    def process_terminate_signal(self) -> ProcessEventClass:
        if not self._process_terminate_signal:
            self._process_terminate_signal = ProcessEvent()
        return self._process_terminate_signal

    @property
    def thread_terminate_signal(self) -> ThreadEvent:
        if not self._thread_terminate_signal:
            self._thread_terminate_signal = ThreadEvent()
        return self._thread_terminate_signal

    @property
    def embeddings(self) -> Embeddings:
        if self.local_embeddings:
            print("Using local embeddings")
            return self.local_embeddings
        print("Using openai embeddings")
        return self.openai_embeddings
