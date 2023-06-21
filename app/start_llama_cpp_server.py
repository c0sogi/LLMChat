from contextlib import asynccontextmanager
import json
import multiprocessing
import time
from functools import partial
from os import path
from typing import Union, Optional, Iterator

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.models.llms import LlamaCppModel, LLMModels

PORT = 8002
ANSI_COLORS = {
    "black": "\u001b[30m",
    "red": "\u001b[31m",
    "green": "\u001b[32m",
    "yellow": "\u001b[33m",
    "blue": "\u001b[34m",
    "magenta": "\u001b[35m",
    "cyan": "\u001b[36m",
    "white": "\u001b[37m",
    "reset": "\u001b[0m",
}


def cprint(text: str, color: str = "reset", **kwargs) -> None:
    current_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(
        f"{ANSI_COLORS[color]}[{current_datetime}] {text}{ANSI_COLORS['reset']}\n",
        **kwargs,
    )


llama_clients: dict = {}
embedding_models: dict = {}
embedding_tokenizers: dict = {}


class CreateEmbeddingRequest(BaseModel):
    model: str = Field(description="The model to use for embedding.")
    input: Union[str, list[str]] = Field(description="The input to embed.")
    user: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "model": "llama_7b",
                "input": "The food was delicious and the waiter...",
            },
        }


def get_embeddings_and_num_of_tokens(
    pretrained_name: str,
    input_texts: list[str],
    context_length: int = 512,
) -> tuple[list[list[float]], int]:
    from gc import collect

    from torch import Tensor
    from transformers.modeling_outputs import (
        BaseModelOutputWithPoolingAndCrossAttentions,
    )
    from transformers.models.auto.modeling_auto import AutoModel
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.models.t5.modeling_t5 import T5Model
    from transformers.tokenization_utils_base import BatchEncoding

    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    if pretrained_name not in embedding_tokenizers:
        # If some other tokenizer exists, delete it
        if embedding_tokenizers:
            del embedding_tokenizers[list(embedding_tokenizers.keys())[0]]
            collect()

        embedding_tokenizers[pretrained_name] = AutoTokenizer.from_pretrained(
            pretrained_name
        )
        cprint(
            f"🤖 Huggingface tokenizer {ANSI_COLORS['white']}{pretrained_name}{ANSI_COLORS['green']} loaded!",
            "green",
        )
    if pretrained_name not in embedding_models:
        if embedding_models:
            del embedding_models[list(embedding_models.keys())[0]]
            collect()
        embedding_models[pretrained_name] = AutoModel.from_pretrained(pretrained_name)
        cprint(
            f"🤖 Huggingface model {ANSI_COLORS['white']}{pretrained_name}{ANSI_COLORS['green']} loaded!",
            "green",
        )

    # Tokenize the input texts
    batch_dict: BatchEncoding = embedding_tokenizers[pretrained_name](
        input_texts,
        max_length=context_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    if isinstance(embedding_models[pretrained_name], T5Model):
        encoder = embedding_models[pretrained_name].get_encoder()
    else:
        encoder = embedding_models[pretrained_name]
    outputs: BaseModelOutputWithPoolingAndCrossAttentions = encoder(**batch_dict)
    return (
        average_pool(
            last_hidden_states=outputs.last_hidden_state,
            attention_mask=batch_dict["attention_mask"],  # type: ignore
        ).tolist(),
        sum([len(encoding) for encoding in batch_dict["input_ids"]]),  # type: ignore
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    cprint("🦙 Llama.cpp server is running", "green")
    yield
    cprint("🦙 Shutting down llama.cpp server...", "red")


def create_app():
    ### ============= Importing llama-cpp-python ============= ###
    try:
        from repositories.llama_cpp.llama_cpp.server import app as llama_cpp_server
        from repositories.llama_cpp.llama_cpp.server.app import llama_cpp

        print("🦙 llama-cpp-python repository found!")
    except Exception as e:
        cprint(
            "🦙 Could not import llama-cpp-python repository. "
            "Please pull the repository with: "
            "git submodule update --recursive\n"
            f"[[[ Received error: {e} ]]]\n"
            f"In {__file__} with name {__name__}\n",
            "yellow",
        )
        from llama_cpp.server import app as llama_cpp_server
        from llama_cpp.server.app import llama_cpp

    class CreateCompletionRequest(llama_cpp_server.CreateCompletionRequest):  # type: ignore
        model: str = Field(default=..., description="The model to use for completion.")

        class Config:
            schema_extra = {
                "example": {
                    "model": "llama_7b",
                    "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                    "stop": ["\n", "###"],
                }
            }

    class CreateChatCompletionRequest(llama_cpp_server.CreateChatCompletionRequest):  # type: ignore
        model: str = Field(default=..., description="The model to use for completion.")

        class Config:
            schema_extra = {
                "example": {
                    "model": "llama_7b",
                    "messages": [
                        llama_cpp_server.ChatCompletionRequestMessage(
                            role="system", content="You are a helpful assistant."
                        ),
                        llama_cpp_server.ChatCompletionRequestMessage(
                            role="user", content="What is the capital of France?"
                        ),
                    ],
                }
            }

    ### ============= Importing llama-cpp-python ============= ###
    router = APIRouter()
    app = FastAPI(
        title="🦙 llama.cpp Python API",
        version="0.0.1",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def convert_llama_cpp_model_to_settings(
        llama_cpp_model: LlamaCppModel,
    ) -> llama_cpp_server.Settings:
        n_threads = (
            max(multiprocessing.cpu_count() // 2, 1)
            if llama_cpp_model.n_threads is None
            else llama_cpp_model.n_threads
        )
        cache_type = "ram" if llama_cpp_model.cache_type is None else "disk"
        cache_size = (
            2 << 30
            if llama_cpp_model.cache_size is None
            else llama_cpp_model.cache_size
        )
        directory, filename = path.split(llama_cpp_model.model_path)
        if not directory:
            directory = "./llama_models/ggml"
        llama_cpp_model.model_path = path.join(directory, filename)

        return llama_cpp_server.Settings(
            model=llama_cpp_model.model_path,
            model_alias=llama_cpp_model.name,
            n_ctx=llama_cpp_model.max_total_tokens,
            n_gpu_layers=llama_cpp_model.n_gpu_layers,
            n_batch=llama_cpp_model.n_batch,
            n_threads=n_threads,
            f16_kv=llama_cpp_model.f16_kv,
            use_mlock=llama_cpp_model.use_mlock,
            use_mmap=llama_cpp_model.use_mmap,
            embedding=llama_cpp_model.embedding,
            low_vram=llama_cpp_model.low_vram,
            last_n_tokens_size=llama_cpp_model.last_n_tokens_size,
            logits_all=llama_cpp_model.logits_all,
            cache=llama_cpp_model.cache,
            cache_type=cache_type,
            cache_size=cache_size,
            vocab_only=llama_cpp_model.vocab_only,
            verbose=llama_cpp_model.echo,
            host="localhost",
            port=PORT,
        )

    def get_settings(
        body: CreateChatCompletionRequest | CreateCompletionRequest,
    ) -> llama_cpp_server.Settings:
        assert body.model is not None
        llama_cpp_model = LLMModels.get_value(body.model)
        assert isinstance(llama_cpp_model, LlamaCppModel)
        return convert_llama_cpp_model_to_settings(llama_cpp_model)

    def get_llama(
        settings: llama_cpp_server.Settings, cache_only_single_model: bool = True
    ) -> llama_cpp.Llama:
        assert settings.model_alias is not None
        if cache_only_single_model:
            # Free all cached models except the current one
            for cached_model_name in [
                key for key in llama_clients.keys() if key != settings.model_alias
            ]:
                client = llama_clients[cached_model_name]
                assert isinstance(client, llama_cpp.Llama)
                if client.ctx is not None:
                    llama_cpp.llama_free(client.ctx)
                client.set_cache(None)
                cprint(f"🦙 Freed LlamaCpp model: {cached_model_name}", "green")
        if settings.model_alias not in llama_clients:
            # Load the current model
            cprint(f"🦙 Loading LlamaCpp model: {settings.model_alias}", "green")
            client = llama_cpp.Llama(
                model_path=settings.model,
                n_gpu_layers=settings.n_gpu_layers,
                f16_kv=settings.f16_kv,
                use_mlock=settings.use_mlock,
                use_mmap=settings.use_mmap,
                embedding=settings.embedding,
                logits_all=settings.logits_all,
                n_threads=settings.n_threads,
                n_batch=settings.n_batch,
                n_ctx=settings.n_ctx,
                last_n_tokens_size=settings.last_n_tokens_size,
                vocab_only=settings.vocab_only,
                verbose=settings.verbose,
            )
            if settings.cache:
                if settings.cache_type == "disk":
                    if settings.verbose:
                        cprint(
                            f"🦙 Using disk cache with size {settings.cache_size}",
                            "green",
                        )
                    cache = llama_cpp.LlamaDiskCache(capacity_bytes=settings.cache_size)
                else:
                    if settings.verbose:
                        cprint(
                            f"🦙 Using ram cache with size {settings.cache_size}",
                            "green",
                        )
                    cache = llama_cpp.LlamaRAMCache(capacity_bytes=settings.cache_size)

                cache = llama_cpp.LlamaCache(capacity_bytes=settings.cache_size)
                client.set_cache(cache)
            assert isinstance(client, llama_cpp.Llama)
            llama_clients[settings.model_alias] = client
        return llama_clients[settings.model_alias]

    @router.post(
        "/v1/chat/completions",
        response_model=llama_cpp_server.CreateChatCompletionResponse,
    )
    async def create_chat_completion(
        request: Request,
        body: CreateChatCompletionRequest,
    ) -> Union[llama_cpp.ChatCompletion, EventSourceResponse]:
        with llama_cpp_server.llama_lock:
            try:
                llama_client = get_llama(get_settings(body))
                exclude = {
                    "n",
                    "logit_bias",
                    "logit_bias_type",
                    "user",
                }
                kwargs = body.dict(exclude=exclude)

                if body.logit_bias is not None:
                    kwargs["logits_processor"] = llama_cpp.LogitsProcessorList(
                        [
                            llama_cpp_server.make_logit_bias_processor(
                                llama_client,
                                body.logit_bias,
                                body.logit_bias_type,
                            ),
                        ]
                    )

                if body.stream:
                    send_chan, recv_chan = anyio.create_memory_object_stream(10)

                    async def event_publisher(inner_send_chan: MemoryObjectSendStream):
                        async with inner_send_chan:
                            try:
                                iterator: Iterator[
                                    llama_cpp.ChatCompletionChunk
                                ] = await run_in_threadpool(
                                    llama_client.create_chat_completion, **kwargs  # type: ignore
                                )
                                cprint(
                                    "\n[🦙 I'm talking now]", color="cyan", flush=True
                                )
                                async for chat_chunk in iterate_in_threadpool(iterator):
                                    print(
                                        chat_chunk["choices"][0]["delta"].get(
                                            "content", ""
                                        ),
                                        end="",
                                        flush=True,
                                    )
                                    await inner_send_chan.send(
                                        dict(data=json.dumps(chat_chunk))
                                    )
                                    if await request.is_disconnected():
                                        raise anyio.get_cancelled_exc_class()()
                                await inner_send_chan.send(dict(data="[DONE]"))
                            except anyio.get_cancelled_exc_class() as e:
                                with anyio.move_on_after(1, shield=True):
                                    cprint(
                                        f"🦙 Disconnected from client (via refresh/close) {request.client}",
                                        color="magenta",
                                    )
                                    await inner_send_chan.send(dict(closing=True))
                                    raise e
                            finally:
                                cprint(
                                    "\n[🦙 I'm done talking]", color="cyan", flush=True
                                )

                    return EventSourceResponse(
                        recv_chan,
                        data_sender_callable=partial(event_publisher, send_chan),
                    )
                else:
                    completion: llama_cpp.ChatCompletion = await run_in_threadpool(
                        llama_client.create_chat_completion, **kwargs  # type: ignore
                    )
                    return completion
            except OSError:
                exit(1)

    @router.post(
        "/v1/completions",
        response_model=llama_cpp_server.CreateCompletionResponse,
    )
    async def create_completion(
        request: Request,
        body: CreateCompletionRequest,
    ) -> Union[llama_cpp.Completion, EventSourceResponse]:
        with llama_cpp_server.llama_lock:
            try:
                llama_client = get_llama(get_settings(body))
                if isinstance(body.prompt, list):
                    assert len(body.prompt) <= 1
                    body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

                exclude = {
                    "n",
                    "best_of",
                    "logit_bias",
                    "logit_bias_type",
                    "user",
                }
                kwargs = body.dict(exclude=exclude)

                if body.logit_bias is not None:
                    kwargs["logits_processor"] = llama_cpp.LogitsProcessorList(
                        [
                            llama_cpp_server.make_logit_bias_processor(
                                llama_client,
                                body.logit_bias,
                                body.logit_bias_type,
                            ),
                        ]
                    )

                if body.stream:
                    send_chan, recv_chan = anyio.create_memory_object_stream(10)

                    async def event_publisher(inner_send_chan: MemoryObjectSendStream):
                        async with inner_send_chan:
                            try:
                                iterator: Iterator[
                                    llama_cpp.CompletionChunk
                                ] = await run_in_threadpool(
                                    llama_client, **kwargs  # type: ignore
                                )
                                cprint(
                                    "\n[🦙 I'm talking now]", color="cyan", flush=True
                                )
                                async for chunk in iterate_in_threadpool(iterator):
                                    print(
                                        chunk["choices"][0]["text"],
                                        end="",
                                        flush=True,
                                    )
                                    await inner_send_chan.send(
                                        dict(data=json.dumps(chunk))
                                    )
                                    if await request.is_disconnected():
                                        raise anyio.get_cancelled_exc_class()()
                                await inner_send_chan.send(dict(data="[DONE]"))
                            except anyio.get_cancelled_exc_class() as e:
                                with anyio.move_on_after(1, shield=True):
                                    cprint(
                                        f"🦙 Disconnected from client (via refresh/close) {request.client}",
                                        color="magenta",
                                    )
                                    await inner_send_chan.send(dict(closing=True))
                                    raise e
                            finally:
                                cprint(
                                    "\n[🦙 I'm done talking]", color="cyan", flush=True
                                )

                    return EventSourceResponse(
                        recv_chan,
                        data_sender_callable=partial(event_publisher, send_chan),
                    )
                else:
                    completion: llama_cpp.Completion = await run_in_threadpool(
                        llama_client, **kwargs  # type: ignore
                    )
                    return completion
            except OSError:
                exit(1)

    @router.post(
        "/v1/embeddings",
        response_model=llama_cpp_server.CreateEmbeddingResponse,
    )
    async def create_embedding(
        request: CreateEmbeddingRequest,
    ) -> llama_cpp.Embedding:
        with llama_cpp_server.llama_lock:
            assert request.model is not None
            if request.model not in (
                "intfloat/e5-large-v2",
                "hkunlp/instructor-xl",
                "hkunlp/instructor-large",
                "intfloat/e5-base-v2",
                "intfloat/e5-large",
            ):
                llama_cpp_model = LLMModels.get_value(request.model)
                assert isinstance(llama_cpp_model, LlamaCppModel)
                if not llama_cpp_model.embedding:
                    raise HTTPException(
                        status_code=400,
                        detail="Model does not support embeddings",
                    )

                llama_client = get_llama(
                    convert_llama_cpp_model_to_settings(llama_cpp_model)
                )
                return await run_in_threadpool(
                    llama_client.create_embedding, **request.dict(exclude={"user"})
                )
            else:
                embeddings, total_tokens = get_embeddings_and_num_of_tokens(
                    pretrained_name=request.model,
                    input_texts=request.input
                    if isinstance(request.input, list)
                    else [request.input],
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
                    "model": request.model,
                    "usage": {
                        "prompt_tokens": total_tokens,
                        "total_tokens": total_tokens,
                    },
                }

    @router.get("/v1/models", response_model=llama_cpp_server.GetModelResponse)
    async def get_models() -> llama_cpp_server.ModelList:
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

    @router.get("/health")
    async def health():
        return "ok"

    app.include_router(router)
    return app


### ============= SERVER =============


def run() -> None:
    from maintools import initialize_before_launch
    from uvicorn import Config, Server

    initialize_before_launch()

    Server(
        config=Config(
            create_app(),
            host="0.0.0.0",
            port=PORT,
            log_level="warning",
        )
    ).run()


# class EncodingUsage(TypedDict):
#     prompt_tokens: int
#     total_tokens: int


# class EncodingData(TypedDict):
#     index: int
#     object: str
#     encoding: list[int]


# class Encoding(TypedDict):
#     object: Literal["list"]
#     model: str
#     data: list[EncodingData]
#     usage: EncodingUsage


# CreateEncodingResponse = create_model_from_typeddict(Encoding)  # type: ignore

# class CreateEncodingRequest(BaseModel):
#     model: str = Field(
#         description="The model to use for encoding. Enter huggingface repo"
#     )
#     input: Union[str, list[str]] = Field(description="The input to encode.")
#     user: Optional[str]

#     class Config:
#         schema_extra = {
#             "example": {
#                 "model": "junelee/wizard-vicuna-13b",
#                 "input": "The food was delicious and the waiter...",
#             },
#         }

# def get_encodings(
#     pretrained_name: str,
#     input_texts: list[str],
# ) -> list[list[int]]:
#     from transformers.models.auto.tokenization_auto import AutoTokenizer
#     from transformers.tokenization_utils_base import BatchEncoding

#     if pretrained_name not in embedding_tokenizers:
#         embedding_tokenizers[pretrained_name] = AutoTokenizer.from_pretrained(
#             pretrained_name
#         )

#     # Tokenize the input texts
#     batch_dict: BatchEncoding = embedding_tokenizers[pretrained_name](
#         input_texts,
#         padding="longest",
#         truncation=False,
#         return_tensors="pt",
#     )
#     return batch_dict["input_ids"].tolist()  # type: ignore

# @router.post(
#     "/v1/encodings",
#     response_model=CreateEncodingResponse,
# )
# async def encoding(
#     request: CreateEncodingRequest,
# ) -> Encoding:
#     with llama_cpp_server.llama_lock:
#         assert request.model is not None
#         if request.model not in (
#             "intfloat/e5-large-v2",
#             "hkunlp/instructor-xl",
#             "hkunlp/instructor-large",
#             "intfloat/e5-base-v2",
#             "intfloat/e5-large",
#         ):
#             tokenizer = LlamaTokenizer(model_name=request.model)
#             if isinstance(request.input, list):
#                 results: list[list[int]] = await gather(
#                     *[
#                         run_in_threadpool(
#                             tokenizer.encode,
#                             input_text,
#                         )
#                         for input_text in request.input
#                     ]
#                 )
#             else:
#                 results: list[list[int]] = [
#                     await run_in_threadpool(
#                         tokenizer.encode,
#                         request.input,
#                     )
#                 ]
#         else:
#             results: list[list[int]] = get_encodings(
#                 pretrained_name=request.model,
#                 input_texts=request.input
#                 if isinstance(request.input, list)
#                 else [request.input],
#             )
#         total_tokens: int = sum(len(result) for result in results)
#         return {
#             "object": "list",
#             "data": [
#                 {
#                     "index": idx,
#                     "object": "encoding",
#                     "encoding": result,
#                 }
#                 for idx, result in enumerate(results)
#             ],
#             "model": request.model,
#             "usage": {
#                 "prompt_tokens": total_tokens,
#                 "total_tokens": total_tokens,
#             },
#         }


if __name__ == "__main__":
    run()
