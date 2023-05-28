"""This module will be spawned in process pool, so parent process can't its global variables."""
from queue import Queue
from sys import stdout
from threading import Event
from time import sleep
from typing import TYPE_CHECKING, Any, Generator, Iterator
from langchain import LlamaCpp
from llama_cpp import Llama, LlamaCache, llama_free, llama_n_ctx
from pydantic import Field, root_validator


from app.errors.chat_exceptions import ChatBreakException, ChatContinueException, ChatLengthException


if TYPE_CHECKING:
    from app.models.chat_models import UserChatContext
    from app.models.llms import LlamaCppModel


class LlamaCppGpu(LlamaCpp):
    n_gpu_layers: int = Field(30, alias="n_gpu_layers")

    @root_validator()
    def validate_environment(cls, values: dict) -> dict:
        model_path = values["model_path"]
        lora_path = values["lora_path"]
        lora_base = values["lora_base"]
        n_ctx = values["n_ctx"]
        n_parts = values["n_parts"]
        seed = values["seed"]
        f16_kv = values["f16_kv"]
        logits_all = values["logits_all"]
        vocab_only = values["vocab_only"]
        use_mlock = values["use_mlock"]
        n_threads = values["n_threads"]
        n_batch = values["n_batch"]
        use_mmap = values["use_mmap"]
        last_n_tokens_size = values["last_n_tokens_size"]
        n_gpu_layers = values["n_gpu_layers"]

        try:
            from llama_cpp import Llama

            values["client"] = Llama(
                n_gpu_layers=n_gpu_layers,
                model_path=model_path,
                lora_base=lora_base,
                lora_path=lora_path,
                n_ctx=n_ctx,
                n_parts=n_parts,
                seed=seed,
                f16_kv=f16_kv,
                logits_all=logits_all,
                vocab_only=vocab_only,
                use_mlock=use_mlock,
                n_threads=n_threads,
                n_batch=n_batch,
                use_mmap=use_mmap,
                last_n_tokens_size=last_n_tokens_size,
            )
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import llama-cpp-python library. "
                "Please install the llama-cpp-python library t  o "
                "use this embedding model: pip install llama-cpp-python"
            )
        except Exception:
            raise NameError(f"Could not load Llama model from path: {model_path}")

        return values


cached: dict[str, LlamaCppGpu] = {}


def get_stops(s: str) -> list[str]:
    return list({s, s.upper(), s.lower(), s.capitalize()})


def get_fake_response() -> Generator:
    for i in range(10):
        yield {
            "id": "fake_id",
            "object": "text_completion",
            "created": 12345,
            "model": "./fake_models/ggml/fake-llama-13B.ggml.q4_2.bin",
            "choices": [{"text": str(i), "index": 0, "logprobs": None, "finish_reason": None}],
        }
        sleep(1)


def load_llama(
    llama_cpp_model: "LlamaCppModel",
    cache_only_single_model: bool = True,
) -> LlamaCppGpu:
    try:
        model_name: str = llama_cpp_model.name
        if cache_only_single_model:
            # Free all cached models except the current one
            for cached_model_name in [key for key in cached.keys() if key != model_name]:
                stdout.write(f">>> Freeing LlamaCpp model: {cached_model_name}")
                client = cached[cached_model_name].client
                assert isinstance(client, Llama)
                if client.ctx is not None:
                    llama_free(client.ctx)
                client.set_cache(None)
                client.reset()
                cached.pop(cached_model_name)
                stdout.write(f">>> Freed LlamaCpp model: {cached_model_name}")
        if model_name not in cached:
            # Load the current model
            stdout.write(f"Loading LlamaCpp model: {model_name}")
            llama_cpp = get_llama(llama_cpp_model)
            assert isinstance(llama_cpp.client, Llama)
            if llama_cpp_model.cache and llama_cpp_model.cache_size is not None:
                llama_cpp.client.set_cache(
                    LlamaCache(capacity_bytes=llama_cpp_model.cache_size),
                )
            cached[model_name] = llama_cpp
        return cached[model_name]
    except Exception as e:
        raise RuntimeError(f"[Error] Could not load llama model: {e}")


def llama_cpp_generation(
    user_chat_context: "UserChatContext",
    prompt: str,
    max_tokens: int,
    m_queue: Queue,  # multiprocessing.managers.AutoProxy[Queue]
    m_done: Event,  # multiprocessing.managers.EventProxy
    is_fake: bool = False,
) -> None:
    def get_generator() -> Iterator[Any]:
        if echo:
            stdout.write("[Info] Creating Response of prompt below:")
            stdout.write(str(prompt))
        assert isinstance(prompt, str)
        if not is_fake:
            llm: LlamaCppGpu = load_llama(llama_cpp_model=llama_cpp_model)
            llm_client: Llama = llm.client
            llm_client.verbose = echo
            real_max_tokens: int = max_tokens

            prompt_tokens: int = len(llm_client.tokenize(b" " + prompt.encode("utf-8")))
            ctx_tokens: int = getattr(llm_client, "_n_ctx", llama_n_ctx(llm_client.ctx))  # type: ignore
            assert isinstance(ctx_tokens, int)
            if prompt_tokens + max_tokens > ctx_tokens:
                stdout.write(
                    (
                        f"\n[Warning] Prompt tokens ({prompt_tokens}) + max tokens ({max_tokens})"
                        f" > n_ctx ({ctx_tokens}). \n"
                    )
                )
                real_max_tokens = ctx_tokens - prompt_tokens
            else:
                real_max_tokens = max_tokens
            if real_max_tokens <= 0:
                raise ChatLengthException()
            return llm_client.create_completion(  # type: ignore
                prompt=prompt,
                suffix=llm.suffix,
                max_tokens=real_max_tokens,
                temperature=user_chat_context.user_chat_profile.temperature,
                top_p=user_chat_context.user_chat_profile.top_p,
                logprobs=llm.logprobs,
                echo=echo,
                stop=llm.stop + avoids if llm.stop is not None else avoids,
                repeat_penalty=user_chat_context.user_chat_profile.frequency_penalty,
                top_k=40,
                stream=True,
            )

        else:
            return get_fake_response()

    try:
        retry_count: int = 0
        content_buffer: str = ""
        llama_cpp_model: "LlamaCppModel" = user_chat_context.llm_model.value  # type: ignore
        echo: bool = llama_cpp_model.echo
        avoids: list[str] = get_stops(
            user_chat_context.user_chat_roles.user + ":",
        ) + get_stops(user_chat_context.user_chat_roles.ai + ":")

        while True:
            try:
                retry_count += 1
                for generation in get_generator():
                    if m_done.is_set():
                        raise InterruptedError("Interruption from main process")
                    elif retry_count > 10:
                        raise RecursionError(f"Exceeded retry limit {retry_count}")
                    finish_reason: str | None = generation["choices"][0]["finish_reason"]  # type: ignore
                    text: str | None = generation["choices"][0].get("text")  # type: ignore

                    if finish_reason == "length":
                        raise ChatLengthException(msg=content_buffer)  # raise exception for token limit
                    if text is not None:
                        if content_buffer == "":
                            text = text.replace("\u200b", "").lstrip()
                        if text != "":
                            if echo:
                                stdout.write(text)
                            content_buffer += text
                            m_queue.put(text)
                if content_buffer.replace("\u200b", "").strip() == "":
                    stdout.write("[Warning] Empty model output. Retrying...")
                    raise ChatContinueException(msg="Empty model output")  # raise exception for empty output
            except ValueError as e:
                if "tokens exceed context window" in str(e):
                    stdout.write("[Warning] Token limit exceeded. Retrying...")
                    raise ChatLengthException(msg=content_buffer)  # raise exception for token limit
                else:
                    stdout.write("[Warning] ValueError. Retrying...")
                    raise e
            except ChatBreakException:
                break
            except ChatContinueException:
                continue
            else:
                break

    except Exception as e:
        m_queue.put_nowait(e)
    else:
        m_queue.put_nowait(None)  # put None to indicate that the process is done


def get_llama(llama_cpp_model: "LlamaCppModel") -> LlamaCppGpu:
    return LlamaCppGpu(
        client=None,
        cache=None,
        callbacks=None,
        callback_manager=None,
        n_gpu_layers=llama_cpp_model.n_gpu_layers,
        model_path=llama_cpp_model.model_path,
        lora_base=llama_cpp_model.lora_base,
        lora_path=llama_cpp_model.lora_path,
        n_ctx=llama_cpp_model.max_total_tokens,
        n_parts=llama_cpp_model.n_parts,
        seed=llama_cpp_model.seed,
        f16_kv=llama_cpp_model.f16_kv,
        logits_all=llama_cpp_model.logits_all,
        vocab_only=llama_cpp_model.vocab_only,
        use_mlock=llama_cpp_model.use_mlock,
        n_threads=llama_cpp_model.n_threads,
        n_batch=llama_cpp_model.n_batch,
        suffix=llama_cpp_model.suffix,
        max_tokens=llama_cpp_model.max_tokens_per_request,
        temperature=llama_cpp_model.temperature,
        top_p=llama_cpp_model.top_p,
        logprobs=llama_cpp_model.logprobs,
        echo=llama_cpp_model.echo,
        stop=llama_cpp_model.stop,
        repeat_penalty=llama_cpp_model.repeat_penalty,
        top_k=llama_cpp_model.top_k,
        last_n_tokens_size=llama_cpp_model.last_n_tokens_size,
        use_mmap=llama_cpp_model.use_mmap,
        streaming=llama_cpp_model.streaming,
    )


class LlamaTokenizerAdapter:
    @staticmethod
    def encode(text: str, llama_cpp_model: "LlamaCppModel") -> list[int]:
        llama = load_llama(llama_cpp_model=llama_cpp_model)
        assert isinstance(llama.client, Llama)
        return llama.client.tokenize(b" " + text.encode("utf-8"))
