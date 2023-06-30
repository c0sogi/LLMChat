import json
from collections import deque
from functools import partial
from typing import Iterator, Union

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException
from pydantic import create_model_from_typeddict
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.models.base_models import (
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
)
from app.models.completion_models import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
    Embedding,
    ModelList,
)
from app.models.llms import ExllamaModel, LlamaCppModel, LLMModel, LLMModels
from app.utils.chat.text_generations import BaseCompletionGenerator
from app.utils.logger import ApiLogger

logger = ApiLogger("||v1||")
try:
    from app.utils.chat.text_generations.llama_cpp.generator import (
        LlamaCppCompletionGenerator,
    )

    logger.cinfo("🦙 Successfully imported llama.cpp module!")
except ImportError as e:
    logger.cwarning(str(e))
    LlamaCppCompletionGenerator = str(e)

try:
    from app.utils.chat.text_generations.exllama.generator import (
        ExllamaCompletionGenerator,
    )

    logger.cinfo("🦙 Successfully imported exllama module!")
except ImportError as e:
    logger.cwarning(str(e))
    ExllamaCompletionGenerator = str(e)
try:
    from app.utils.chat.embeddings import get_embeddings_and_num_of_tokens

    logger.cinfo("🦙 Successfully imported embeddings(Pytorch + Transformer) module!")
except ImportError as e:
    logger.cwarning(str(e))
    get_embeddings_and_num_of_tokens = str(e)


router = APIRouter()

embedding_models: dict = {}
embedding_tokenizers: dict = {}

completion_generators: deque[BaseCompletionGenerator] = deque(maxlen=1)
semaphore = anyio.create_semaphore(1)


async def get_semaphore():
    async with semaphore:
        yield semaphore


async def get_exception_handler():
    try:
        yield
    except Exception as e:
        logger.error(f"Exception in llama-cpp: {e}")
        if isinstance(e, OSError):
            exit(1)
        if isinstance(e, AssertionError):
            raise HTTPException(status_code=400, detail={"error": str(e)})
        raise HTTPException(status_code=500, detail={"error": "Internal Server Error"})


def get_completion_generator(
    body: CreateCompletionRequest
    | CreateChatCompletionRequest
    | CreateEmbeddingRequest,
) -> BaseCompletionGenerator:
    try:
        llm_model: LLMModel = LLMModels.get_value(body.model)
        for completion_generator in completion_generators:
            if completion_generator.llm_model is llm_model:
                return completion_generator
        if isinstance(llm_model, LlamaCppModel):
            if isinstance(LlamaCppCompletionGenerator, str):
                raise AssertionError(LlamaCppCompletionGenerator)
            completion_generators.append(
                LlamaCppCompletionGenerator.from_pretrained(llm_model)
            )
            return completion_generators[-1]
        elif isinstance(llm_model, ExllamaModel):
            if isinstance(ExllamaCompletionGenerator, str):
                raise AssertionError(ExllamaCompletionGenerator)
            completion_generators.append(
                ExllamaCompletionGenerator.from_pretrained(llm_model)
            )
            return completion_generators[-1]
        else:
            raise AssertionError(f"Model {body.model} not implemented")
    except AssertionError as e:
        raise e
    except Exception:
        raise AssertionError(f"Could not find a model: {body.model}")


@router.post(
    "/v1/chat/completions",
    response_model=create_model_from_typeddict(ChatCompletion),  # type: ignore
)
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
    semaphore: None = Depends(get_semaphore),
    exception_handler: None = Depends(get_exception_handler),
) -> Union[ChatCompletion, EventSourceResponse]:
    completion_generator = get_completion_generator(body)
    if body.stream:
        send_chan, recv_chan = anyio.create_memory_object_stream(10)

        async def event_publisher(inner_send_chan: MemoryObjectSendStream):
            async with inner_send_chan:
                try:
                    iterator: Iterator[ChatCompletionChunk] = await run_in_threadpool(
                        completion_generator.generate_chat_completion_with_streaming,
                        messages=body.messages,
                        settings=body,
                    )
                    logger.info("\n[🦙 I'm talking now]")
                    async for chat_chunk in iterate_in_threadpool(iterator):
                        print(
                            chat_chunk["choices"][0]["delta"].get("content", ""),
                            end="",
                            flush=True,
                        )
                        await inner_send_chan.send(dict(data=json.dumps(chat_chunk)))
                        if await request.is_disconnected():
                            raise anyio.get_cancelled_exc_class()()
                    await inner_send_chan.send(dict(data="[DONE]"))
                except anyio.get_cancelled_exc_class() as e:
                    with anyio.move_on_after(1, shield=True):
                        logger.info(
                            f"🦙 Disconnected from client (via refresh/close) {request.client}",
                        )
                        await inner_send_chan.send(dict(closing=True))
                        raise e
                finally:
                    logger.info("\n[🦙 I'm done talking]")

        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(event_publisher, send_chan),
        )
    else:
        chat_completion: ChatCompletion = await run_in_threadpool(
            completion_generator.generate_chat_completion,
            messages=body.messages,
            settings=body,
        )
        return chat_completion


@router.post(
    "/v1/completions",
    response_model=create_model_from_typeddict(Completion),  # type: ignore
)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
    semaphore: None = Depends(get_semaphore),
    exception_handler: None = Depends(get_exception_handler),
) -> Union[Completion, EventSourceResponse]:
    logger.info(f"🦙 {body.prompt}")
    completion_generator = get_completion_generator(body)
    if body.stream:
        send_chan, recv_chan = anyio.create_memory_object_stream(10)

        async def event_publisher(inner_send_chan: MemoryObjectSendStream):
            async with inner_send_chan:
                try:
                    iterator: Iterator[CompletionChunk] = await run_in_threadpool(
                        completion_generator.generate_completion_with_streaming,
                        prompt=body.prompt,
                        settings=body,
                    )
                    logger.info("\n[🦙 I'm talking now]")
                    async for chunk in iterate_in_threadpool(iterator):
                        print(
                            chunk["choices"][0]["text"],
                            end="",
                            flush=True,
                        )
                        await inner_send_chan.send(dict(data=json.dumps(chunk)))
                        if await request.is_disconnected():
                            raise anyio.get_cancelled_exc_class()()
                    await inner_send_chan.send(dict(data="[DONE]"))
                except anyio.get_cancelled_exc_class() as e:
                    with anyio.move_on_after(1, shield=True):
                        logger.info(
                            f"🦙 Disconnected from client (via refresh/close) {request.client}",
                        )
                        await inner_send_chan.send(dict(closing=True))
                        raise e
                finally:
                    logger.info("\n[🦙 I'm done talking!]")

        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(event_publisher, send_chan),
        )
    else:
        completion: Completion = await run_in_threadpool(
            completion_generator.generate_completion,
            prompt=body.prompt,
            settings=body,
        )
        return completion


@router.post(
    "/v1/embeddings",
    response_model=create_model_from_typeddict(Embedding),  # type: ignore
)
async def create_embedding(
    body: CreateEmbeddingRequest,
    semaphore: None = Depends(get_semaphore),
    exception_handler: None = Depends(get_exception_handler),
) -> Embedding:
    assert body.model is not None, "Model is required"
    if body.model in (
        "intfloat/e5-large-v2",
        "hkunlp/instructor-xl",
        "hkunlp/instructor-large",
        "intfloat/e5-base-v2",
        "intfloat/e5-large",
    ):
        # Embedding model from Transformer
        assert not isinstance(
            get_embeddings_and_num_of_tokens, str
        ), get_embeddings_and_num_of_tokens

        embeddings, total_tokens = get_embeddings_and_num_of_tokens(
            pretrained_name=body.model,
            input_texts=body.input if isinstance(body.input, list) else [body.input],
            context_length=512,
        )

        return {
            "object": "list",
            "data": [
                {
                    "index": idx,
                    "object": "embedding",
                    "embedding": embedding,
                }
                for idx, embedding in enumerate(embeddings)
            ],
            "model": body.model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }
    else:
        # Embedding model from Llama.cpp
        try:
            from app.utils.chat.text_generations.llama_cpp.generator import (
                LlamaCppCompletionGenerator,
            )
        except ImportError as e:
            raise AssertionError(
                f"Cannot import required libraries for embeddings: {e}"
            )

        llama_cpp_model = LLMModels.get_value(body.model)
        assert isinstance(
            llama_cpp_model, LlamaCppModel
        ), "Non-llama-cpp model is not supported"
        assert (
            llama_cpp_model.embedding
        ), "Model does not support embeddings. Set `embedding` to True in the LlamaCppModel"

        completion_generator = get_completion_generator(body)
        assert isinstance(
            completion_generator, LlamaCppCompletionGenerator
        ), "Non-llama-cpp model is not supported"
        assert completion_generator.client, "Model is not loaded yet"
        return await run_in_threadpool(
            completion_generator.client.create_embedding, **body.dict(exclude={"user"})
        )


@router.get("/v1/models", response_model=create_model_from_typeddict(Embedding))  # type: ignore
async def get_models() -> ModelList:
    llama_cpp_models: list[LlamaCppModel] = [
        enum.value
        for enum in LLMModels._member_map_.values()
        if isinstance(enum.value, LlamaCppModel)
    ]
    return {
        "object": "list",
        "data": [
            {
                "id": llama_cpp_model.name + f"({llama_cpp_model.model_path})",
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
            for llama_cpp_model in llama_cpp_models
        ],
    }
