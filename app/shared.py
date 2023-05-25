from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Manager
from multiprocessing.managers import SyncManager

import openai
from app.common.config import OPENAI_API_KEY, SingletonMetaClass
from langchain.embeddings import OpenAIEmbeddings


@dataclass
class Shared(metaclass=SingletonMetaClass):
    process_manager: SyncManager = field(default_factory=Manager)
    process_pool_executor: ProcessPoolExecutor = field(default_factory=ProcessPoolExecutor)
    openai_embeddings: OpenAIEmbeddings = field(
        default_factory=lambda: OpenAIEmbeddings(client=openai.Embedding, openai_api_key=OPENAI_API_KEY)
    )
