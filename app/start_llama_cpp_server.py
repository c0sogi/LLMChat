import time
from threading import Event
from typing import Optional


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


def create_app():
    ### ============= Imports ============= ###

    import json
    import multiprocessing

    from functools import partial
    from os import path

    from typing import Iterator, Union

    import anyio
    from anyio.streams.memory import MemoryObjectSendStream
    from fastapi import APIRouter, FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.exceptions import HTTPException
    from sse_starlette.sse import EventSourceResponse
    from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

    from app.models.llms import LlamaCppModel, LLMModels

    ### ============= Importing llama-cpp-python ============= ###
    try:
        from repositories.llama_cpp.llama_cpp.server import app as llama_cpp_server
        from repositories.llama_cpp.llama_cpp.server.app import llama_cpp
    except Exception as e:
        if isinstance(e, ImportError) and __name__ == "__main__":
            cprint(
                "🦙 Could not find llama-cpp-python repository. "
                "Please pull the repository with: "
                "git submodule update --recursive\n"
                f"[[[ Received error: {e} ]]]\n"
                f"In {__file__} with name {__name__}\n",
                "yellow",
            )

        elif __name__ == "__main__":
            cprint(
                "🦙 Some other error occurred while importing llama-cpp-python repository.\n"
                f"[[[ Received error: {e} ]]]\n"
                f"In {__file__} with name {__name__}\n",
                "yellow",
            )
        from llama_cpp.server import app as llama_cpp_server
        from llama_cpp.server.app import llama_cpp

    ### ============= Main code ============= ###

    cached: dict[str, llama_cpp.Llama] = {}
    router = APIRouter()
    app = FastAPI(
        title="🦙 llama.cpp Python API",
        version="0.0.1",
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
            port=8002,
        )

    def get_settings(
        body: llama_cpp_server.CreateChatCompletionRequest,
    ) -> llama_cpp_server.Settings:
        assert body.model is not None
        llama_cpp_model = LLMModels._member_map_[body.model].value
        assert isinstance(llama_cpp_model, LlamaCppModel)
        return convert_llama_cpp_model_to_settings(llama_cpp_model)

    def get_llama_with_lock(
        settings: llama_cpp_server.Settings, cache_only_single_model: bool = True
    ) -> llama_cpp.Llama:
        assert settings.model_alias is not None
        with llama_cpp_server.llama_lock:
            if cache_only_single_model:
                # Free all cached models except the current one
                for cached_model_name in [
                    key for key in cached.keys() if key != settings.model_alias
                ]:
                    client = cached[cached_model_name]
                    assert isinstance(client, llama_cpp.Llama)
                    if client.ctx is not None:
                        llama_cpp.llama_free(client.ctx)
                    client.set_cache(None)
                    cprint(f"🦙 Freed LlamaCpp model: {cached_model_name}", "green")
            if settings.model_alias not in cached:
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
                        cache = llama_cpp.LlamaDiskCache(
                            capacity_bytes=settings.cache_size
                        )
                    else:
                        if settings.verbose:
                            cprint(
                                f"🦙 Using ram cache with size {settings.cache_size}",
                                "green",
                            )
                        cache = llama_cpp.LlamaRAMCache(
                            capacity_bytes=settings.cache_size
                        )

                    cache = llama_cpp.LlamaCache(capacity_bytes=settings.cache_size)
                    client.set_cache(cache)  # type: ignore
                assert isinstance(client, llama_cpp.Llama)
                cached[settings.model_alias] = client
            return cached[settings.model_alias]

    @router.post(
        "/v1/chat/completions",
        response_model=llama_cpp_server.CreateChatCompletionResponse,
    )
    async def create_chat_completion(
        request: Request,
        body: llama_cpp_server.CreateChatCompletionRequest,
    ) -> Union[llama_cpp.ChatCompletion, EventSourceResponse]:
        try:
            llama_client = get_llama_with_lock(get_settings(body))  # type: ignore
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
                            llama_client,  # type: ignore
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
        body: llama_cpp_server.CreateCompletionRequest,
    ) -> Union[llama_cpp.Completion, EventSourceResponse]:
        try:
            llama_client = get_llama_with_lock(get_settings(body))  # type: ignore
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
                            llama_client,  # type: ignore
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
                                cprint(
                                    f"🦙 Disconnected from client (via refresh/close) {request.client}",
                                    color="magenta",
                                )
                                await inner_send_chan.send(dict(closing=True))
                                raise e

                return EventSourceResponse(
                    recv_chan, data_sender_callable=partial(event_publisher, send_chan)
                )
            else:
                completion: llama.Completion = await run_in_threadpool(llama_client, **kwargs)  # type: ignore
                return completion
        except OSError:
            exit(1)

    @router.post(
        "/v1/embeddings",
        response_model=llama_cpp_server.CreateEmbeddingResponse,
    )
    async def create_embedding(
        request: llama_cpp_server.CreateEmbeddingRequest,
    ):
        assert request.model is not None
        llama_cpp_model = LLMModels._member_map_[request.model].value
        assert isinstance(llama_cpp_model, LlamaCppModel)
        if not llama_cpp_model.embedding:
            raise HTTPException(
                status_code=400,
                detail="Model does not support embeddings",
            )

        llama_client = get_llama_with_lock(
            convert_llama_cpp_model_to_settings(llama_cpp_model)
        )
        return await run_in_threadpool(
            llama_client.create_embedding, **request.dict(exclude={"user"})
        )

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

    @app.on_event("startup")
    def startup_event():
        cprint("🦙 Llama.cpp server is running!", "green")

    @app.on_event("shutdown")
    def shutdown_event():
        cprint("🦙 Shutting down llama.cpp server...", "red")

    app.include_router(router)
    return app


### ============= SERVER =============


def run(terminate_event: Optional[Event] = None):
    import uvicorn
    from asyncio import get_event_loop, gather, sleep

    if terminate_event is None:
        terminate_event = Event()

    server = uvicorn.Server(
        config=uvicorn.Config(
            create_app(), host="0.0.0.0", port=8002, log_level="warning"
        )
    )

    async def shutdown_when_event_set():
        while not terminate_event.is_set():
            await sleep(0.5)  # sleep for a while before checking again
        await server.shutdown()

    get_event_loop().run_until_complete(
        gather(
            server.serve(),
            shutdown_when_event_set(),
        )
    )


if __name__ == "__main__":
    run()
