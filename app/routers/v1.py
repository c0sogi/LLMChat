"""V1 Endpoints for Local Llama API
Use same format as OpenAI API"""

import json
from collections import deque
from functools import partial
from os import getpid, kill
from signal import SIGINT
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Iterator, Optional, Union

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
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
from app.models.system import free_memory_from_deque
from app.utils.logger import ApiLogger

logger = ApiLogger("||v1||")

# Importing llama.cpp
try:
    from app.utils.chat.text_generations.llama_cpp.generator import (
        LlamaCppCompletionGenerator,
    )

    logger.info("🦙 Successfully imported llama.cpp module!")
except ImportError as e:
    logger.warning(str(e))
    LlamaCppCompletionGenerator = str(e)  # Import error message

# Importing exllama
try:
    from app.utils.chat.text_generations.exllama.generator import (
        ExllamaCompletionGenerator,
    )

    logger.info("🦙 Successfully imported exllama module!")
except ImportError as e:
    logger.warning(str(e))
    ExllamaCompletionGenerator = str(e)  # Import error message


# Importing embeddings (Pytorch + Transformer)
try:
    from app.utils.chat.embeddings import TransformerEmbedding

    logger.info("🦙 Successfully imported embeddings(Pytorch + Transformer) module!")
except ImportError as e:
    logger.warning(str(e))
    TransformerEmbedding = str(e)  # Import error message

if TYPE_CHECKING:
    from app.utils.chat.embeddings import BaseEmbedding
    from app.utils.chat.text_generations import BaseCompletionGenerator

OPENAI_REPLACEMENT_MODELS: dict[str, str] = {
    "gpt-3.5-turbo": "chronos_hermes_13b",
    "gpt-3.5-turbo-16k": "longchat_7b",
    "gpt-4": "pygmalion_13b",
}


class RouteErrorHandler(APIRoute):
    """Custom APIRoute that handles application errors and exceptions"""

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except (OSError, MemoryError) as e:
                logger.exception(f"Exception in llama-cpp: {e}")
                if isinstance(e, MemoryError):
                    error_msg = str(e)
                else:
                    error_msg = "Memory corruption occurred. Terminating..."
                kill(getpid(), SIGINT)
                return JSONResponse({"error": error_msg}, 500)
            except AssertionError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            except Exception as e:
                logger.exception(f"Exception in llama-cpp: {e}")
                return JSONResponse(
                    {"error": "Internal Server Error in V1"}, status_code=500
                )
            finally:
                ...

        return custom_route_handler


router = APIRouter(route_class=RouteErrorHandler)
semaphore = anyio.create_semaphore(1)
completion_generators: deque["BaseCompletionGenerator"] = deque(maxlen=1)
transformer_embeddings: deque["BaseEmbedding"] = deque(maxlen=1)


async def get_semaphore() -> AsyncGenerator[anyio.Semaphore, None]:
    """Get a semaphore for the endpoint. This is to prevent multiple requests from
    creating multiple completion generators at the same time."""
    async with semaphore:
        yield semaphore


def get_completion_generator(
    body: CreateCompletionRequest
    | CreateChatCompletionRequest
    | CreateEmbeddingRequest,
) -> "BaseCompletionGenerator":
    """Get a completion generator for the given model. If the model is not cached, create a new one.
    If the cache is full, delete the oldest completion generator."""
    try:
        # Check if the model is an OpenAI model
        if body.model in OPENAI_REPLACEMENT_MODELS:
            body.model = OPENAI_REPLACEMENT_MODELS[body.model]
            if not isinstance(body, CreateEmbeddingRequest):
                body.logit_bias = None

        # Check if the model is defined in LLMModels enum
        llm_model: LLMModel = LLMModels.get_value(body.model)

        # Check if the model is cached. If so, return the cached completion generator
        for completion_generator in completion_generators:
            if completion_generator.llm_model is llm_model:
                return completion_generator

        # Before creating a new completion generator, check memory usage
        if completion_generators.maxlen == len(completion_generators):
            free_memory_from_deque(
                deque_object=completion_generators,
                min_free_memory_mb=512,
                logger=logger,
            )

        # Create a new completion generator
        if isinstance(llm_model, LlamaCppModel):
            assert not isinstance(
                LlamaCppCompletionGenerator, str
            ), LlamaCppCompletionGenerator
            to_return = LlamaCppCompletionGenerator.from_pretrained(llm_model)
        elif isinstance(llm_model, ExllamaModel):
            assert not isinstance(
                ExllamaCompletionGenerator, str
            ), ExllamaCompletionGenerator
            to_return = ExllamaCompletionGenerator.from_pretrained(llm_model)
        else:
            raise AssertionError(f"Model {body.model} not implemented")

        # Add the new completion generator to the deque cache
        completion_generators.append(to_return)
        return to_return
    except (AssertionError, OSError, MemoryError) as e:
        raise e
    except Exception as e:
        logger.exception(f"Exception in get_completion_generator: {e}")
        raise AssertionError(f"Could not find a model: {body.model}")


def get_transformer_embedding(
    pretrained_name: str,
) -> "BaseEmbedding":
    """Get an embedding for the given model. If the model is not cached, create a new one.
    If the cache is full, delete the oldest completion generator."""
    try:
        for embedding in transformer_embeddings:
            if embedding.model_name == pretrained_name:
                return embedding

        # Before creating a new completion generator, check memory usage
        if transformer_embeddings.maxlen == len(transformer_embeddings):
            free_memory_from_deque(
                deque_object=transformer_embeddings,
                min_free_memory_mb=512,
                logger=logger,
            )

        # Create a new transformer embedding
        assert not isinstance(TransformerEmbedding, str), TransformerEmbedding
        to_return = TransformerEmbedding.from_pretrained(
            pretrained_name=pretrained_name
        )

        # Add the new completion generator to the deque cache
        transformer_embeddings.append(to_return)
        return to_return
    except (AssertionError, OSError, MemoryError) as e:
        raise e
    except Exception as e:
        logger.exception(f"Exception in get_transformer_embedding: {e}")
        raise AssertionError(f"Could not find a model: {pretrained_name}")


@router.post(
    "/v1/chat/completions",
    response_model=create_model_from_typeddict(ChatCompletion),  # type: ignore
)
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
    semaphore: anyio.Semaphore = Depends(get_semaphore),
) -> Union[ChatCompletion, EventSourceResponse]:
    logger.info(f"🦙 Chat Completion Settings: {body}\n\n")
    completion_generator = get_completion_generator(body)
    logger.info("\n[🦙 I'm talking now]")
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
        print(chat_completion["choices"][0]["message"]["content"])
        logger.info("\n[🦙 I'm done talking!]")
        return chat_completion


@router.post(
    "/v1/completions",
    response_model=create_model_from_typeddict(Completion),  # type: ignore
)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
    semaphore: anyio.Semaphore = Depends(get_semaphore),
) -> Union[Completion, EventSourceResponse]:
    logger.info(f"🦙 Text Completion Settings: {body}\n\n")
    completion_generator = get_completion_generator(body)
    logger.info("\n[🦙 I'm talking now]")
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
        print(completion["choices"][0]["text"])
        logger.info("\n[🦙 I'm done talking!]")
        return completion


@router.post(
    "/v1/embeddings",
    response_model=create_model_from_typeddict(Embedding),  # type: ignore
)
async def create_embedding(
    body: CreateEmbeddingRequest,
    semaphore: anyio.Semaphore = Depends(get_semaphore),
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
        transformer_embedding = get_transformer_embedding(body.model)

        (
            embeddings,
            total_tokens,
        ) = transformer_embedding.generate_embeddings_and_n_tokens(
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
        assert not isinstance(
            LlamaCppCompletionGenerator, str
        ), LlamaCppCompletionGenerator
        completion_generator = get_completion_generator(body)
        assert isinstance(
            completion_generator, LlamaCppCompletionGenerator
        ), "Non-llama-cpp model is not supported"
        assert (
            completion_generator.llm_model.embedding
        ), "Model does not support embeddings. Set `embedding` to True in the LlamaCppModel"

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
