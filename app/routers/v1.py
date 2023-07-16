"""V1 Endpoints for Local Llama API
Use same format as OpenAI API"""


from collections import deque
from functools import partial
from typing import TYPE_CHECKING, AsyncGenerator, Iterator, Optional, Union

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import APIRouter, Depends, Request
from orjson import dumps
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
from app.utils.errors import RouteErrorHandler
from app.utils.logger import ApiLogger
from app.utils.module_reloader import ModuleReloader
from app.utils.system import free_memory_of_first_item_from_container

logger = ApiLogger("||v1||")


# Importing llama.cpp
try:
    from app.utils.chat.text_generations.llama_cpp import (
        LlamaCppCompletionGenerator,
    )

    logger.info("🦙 Successfully imported llama.cpp module!")
except Exception as e:
    logger.warning("Llama.cpp import error: " + str(e))
    LlamaCppCompletionGenerator = str(e)  # Import error message


# Importing exllama
try:
    from app.utils.chat.text_generations.exllama import (
        ExllamaCompletionGenerator,
    )

    logger.info("🦙 Successfully imported exllama module!")
except Exception as e:
    logger.exception("Exllama package import error: " + str(e))
    ExllamaCompletionGenerator = str(e)  # Import error message


# Importing embeddings (Pytorch + Transformer)
try:
    from app.utils.chat.embeddings.transformer import (
        TransformerEmbeddingGenerator,
    )

    logger.info(
        "🦙 Successfully imported embeddings(Pytorch + Transformer) module!"
    )
except Exception as e:
    logger.warning("Transformer embedding import error: " + str(e))
    TransformerEmbeddingGenerator = str(e)  # Import error message


# Importing embeddings (Tensorflow + Sentence Encoder)
try:
    from app.utils.chat.embeddings.sentence_encoder import (
        SentenceEncoderEmbeddingGenerator,
    )

    logger.info(
        "🦙 Successfully imported embeddings(Tensorflow + Sentence Encoder) module!"
    )
except Exception as e:
    logger.warning("Sentence Encoder embedding import error: " + str(e))
    SentenceEncoderEmbeddingGenerator = str(e)  # Import error message


if TYPE_CHECKING:
    from app.models.llms import (
        ExllamaModel,
        LlamaCppModel,
        LLMModel,
        LLMModels,
    )
    from app.utils.chat.embeddings import BaseEmbeddingGenerator
    from app.utils.chat.text_generations import BaseCompletionGenerator

OPENAI_REPLACEMENT_MODELS: dict[str, str] = {
    "gpt-3.5-turbo": "chronos_hermes_13b",
    "gpt-3.5-turbo-16k": "longchat_7b",
    "gpt-4": "pygmalion_13b",
}
router = APIRouter(route_class=RouteErrorHandler)
semaphore = anyio.create_semaphore(1)
completion_generators: deque["BaseCompletionGenerator"] = deque(maxlen=1)
embedding_generators: deque["BaseEmbeddingGenerator"] = deque(maxlen=1)


class DynamicLLMS:
    """Dynamically reloads the llms module when it is changed.
    This is to prevent the need to restart the server when the llms module is changed.
    """

    llms_module_reloader: ModuleReloader = ModuleReloader("app.models.llms")

    @classmethod
    def reload(cls):
        return cls.llms_module_reloader.reload()

    @classmethod
    @property
    def llm_models(cls) -> "LLMModels":
        return cls.llms_module_reloader.module.LLMModels

    @classmethod
    @property
    def llama_cpp_model(cls) -> "LlamaCppModel":
        return cls.llms_module_reloader.module.LlamaCppModel

    @classmethod
    @property
    def exllama_model(cls) -> "ExllamaModel":
        return cls.llms_module_reloader.module.ExllamaModel

    @classmethod
    def check_is_llama_cpp_model(cls, llm_model: "LLMModel") -> bool:
        llms = cls.llms_module_reloader.module
        return isinstance(llm_model, llms.LlamaCppModel)

    @classmethod
    def check_is_exllama_model(cls, llm_model: "LLMModel") -> bool:
        llms = cls.llms_module_reloader.module
        return isinstance(llm_model, llms.ExllamaModel)


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
        DynamicLLMS.reload()
        llm_model = DynamicLLMS.llm_models.get_value(body.model)

        # Check if the model is cached. If so, return the cached completion generator
        for completion_generator in completion_generators:
            if completion_generator.llm_model.name == llm_model.name:
                return completion_generator

        # Before creating a new completion generator, deallocate embeddings to free up memory
        if embedding_generators:
            free_memory_of_first_item_from_container(
                embedding_generators,
                min_free_memory_mb=512,
                logger=logger,
            )

        # Before creating a new completion generator, check memory usage
        if completion_generators.maxlen == len(completion_generators):
            free_memory_of_first_item_from_container(
                completion_generators,
                min_free_memory_mb=256,
                logger=logger,
            )

        # Create a new completion generator
        if DynamicLLMS.check_is_llama_cpp_model(llm_model):
            assert not isinstance(
                LlamaCppCompletionGenerator, str
            ), LlamaCppCompletionGenerator
            to_return = LlamaCppCompletionGenerator.from_pretrained(llm_model)
        elif DynamicLLMS.check_is_exllama_model(llm_model):
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


def get_embedding_generator(
    body: CreateEmbeddingRequest,
) -> "BaseEmbeddingGenerator":
    """Get an embedding generator for the given model. If the model is not cached, create a new one.
    If the cache is full, delete the oldest completion generator."""
    try:
        body.model = body.model.lower()
        for embedding_generator in embedding_generators:
            if embedding_generator.model_name == body.model:
                return embedding_generator

        # Before creating a new completion generator, check memory usage
        if embedding_generators.maxlen == len(embedding_generators):
            free_memory_of_first_item_from_container(
                embedding_generators,
                min_free_memory_mb=256,
                logger=logger,
            )
        # Before creating a new completion generator, deallocate embeddings to free up memory
        if completion_generators:
            free_memory_of_first_item_from_container(
                completion_generators,
                min_free_memory_mb=512,
                logger=logger,
            )

        if "sentence" in body.model and "encoder" in body.model:
            # Create a new sentence encoder embedding
            assert not isinstance(
                SentenceEncoderEmbeddingGenerator, str
            ), SentenceEncoderEmbeddingGenerator
            to_return = SentenceEncoderEmbeddingGenerator.from_pretrained(
                body.model
            )
        else:
            # Create a new transformer embedding
            assert not isinstance(
                TransformerEmbeddingGenerator, str
            ), LlamaCppCompletionGenerator
            to_return = TransformerEmbeddingGenerator.from_pretrained(
                body.model
            )

        # Add the new completion generator to the deque cache
        embedding_generators.append(to_return)
        return to_return
    except (AssertionError, OSError, MemoryError) as e:
        raise e
    except Exception as e:
        logger.exception(f"Exception in get_embedding_generator: {e}")
        raise AssertionError(f"Could not find a model: {body.model}")


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Iterator,
    is_chat_completion: Optional[bool] = None,
):
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                if is_chat_completion is True:
                    print(
                        chunk["choices"][0]["delta"].get("content", ""),
                        end="",
                        flush=True,
                    )
                elif is_chat_completion is False:
                    print(
                        chunk["choices"][0]["text"],
                        end="",
                        flush=True,
                    )
                await inner_send_chan.send(b"data: " + dumps(chunk) + b"\n\n")
                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(b"data: [DONE]\n\n")
        except anyio.get_cancelled_exc_class() as e:
            with anyio.move_on_after(1, shield=True):
                logger.info(
                    f"🦙 Disconnected from client (via refresh/close) {request.client}",
                )
                raise e
        finally:
            logger.info("\n[🦙 I'm done talking]")


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
        _iterator: Iterator[
            ChatCompletionChunk
        ] = completion_generator.generate_chat_completion_with_streaming(
            messages=body.messages,
            settings=body,
        )
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, _iterator)

        def iterator() -> Iterator[ChatCompletionChunk]:
            yield first_response
            yield from _iterator

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
                is_chat_completion=True,
            ),
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
        _iterator: Iterator[
            CompletionChunk
        ] = completion_generator.generate_completion_with_streaming(
            prompt=body.prompt,
            settings=body,
        )
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, _iterator)

        def iterator() -> Iterator[CompletionChunk]:
            yield first_response
            yield from _iterator

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
                is_chat_completion=False,
            ),
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
    try:
        DynamicLLMS.reload()
        llm_model = DynamicLLMS.llm_models.get_value(body.model)
        if not DynamicLLMS.check_is_llama_cpp_model(llm_model):
            raise NotImplementedError("Using non-llama-cpp model")

    except Exception:
        # Embedding model from local
        #     "intfloat/e5-large-v2",
        #     "hkunlp/instructor-xl",
        #     "hkunlp/instructor-large",
        #     "intfloat/e5-base-v2",
        #     "intfloat/e5-large",
        embedding_generator: "BaseEmbeddingGenerator" = (
            get_embedding_generator(body)
        )
        embeddings: list[list[float]] = await run_in_threadpool(
            embedding_generator.generate_embeddings,
            texts=body.input if isinstance(body.input, list) else [body.input],
            context_length=512,
            batch=1000,
        )

        return {
            "object": "list",
            "data": [
                {
                    "index": embedding_idx,
                    "object": "embedding",
                    "embedding": embedding,
                }
                for embedding_idx, embedding in enumerate(embeddings)
            ],
            "model": body.model,
            "usage": {
                "prompt_tokens": -1,
                "total_tokens": -1,
            },
        }

    else:
        # Trying to get embedding model from Llama.cpp
        assert (
            llm_model.embedding
        ), "Model does not support embeddings. Set `embedding` to True in the LlamaCppModel"
        assert not isinstance(
            LlamaCppCompletionGenerator, str
        ), LlamaCppCompletionGenerator
        completion_generator = get_completion_generator(body)
        assert isinstance(
            completion_generator, LlamaCppCompletionGenerator
        ), f"Model {body.model} is not supported for llama.cpp embeddings."

        assert completion_generator.client, "Model is not loaded yet"
        return await run_in_threadpool(
            completion_generator.client.create_embedding,
            **body.dict(exclude={"user"}),
        )


@router.get("/v1/models", response_model=create_model_from_typeddict(ModelList))  # type: ignore
async def get_models() -> ModelList:
    DynamicLLMS.reload()
    llama_cpp_models: list = [
        enum.value for enum in DynamicLLMS.llm_models.member_map.values()
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
