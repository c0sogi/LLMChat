"""This module will be spawned in process pool, so parent process can't its global variables."""
import logging
from typing import TYPE_CHECKING, Generator

from langchain import LlamaCpp
from llama_cpp import Llama, LlamaCache, llama_free
from pydantic import Field, root_validator


from app.errors.chat_exceptions import ChatBreakException, ChatContinueException, ChatLengthException
from app.utils.chat.chat_config import ChatConfig

if TYPE_CHECKING:
    from app.models.chat_models import UserChatContext
    from app.models.llms import LlamaCppModel

logger = logging.getLogger(__name__)


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
                "Please install the llama-cpp-python library to "
                "use this embedding model: pip install llama-cpp-python"
            )
        except Exception:
            raise NameError(f"Could not load Llama model from path: {model_path}")

        return values


LLAMA_CACHED: dict[str, LlamaCppGpu] = {}


def can_avoid_in_buffer(text_buffer: str, text: str, avoids: list[str]) -> bool:
    for avoid in avoids:
        avoid = avoid.upper()
        possible_buffer = (text_buffer + text).upper()
        if avoid in possible_buffer or any([possible_buffer.endswith(avoid[: i + 1]) for i in range(len(avoid))]):
            return True
    return False


def get_stops(s: str) -> list[str]:
    return [
        s,
        s.upper(),
        s.lower(),
        s.capitalize(),
    ]


def get_fake_response() -> Generator:
    from random import randint

    yield {
        "id": "fake_id",
        "object": "text_completion",
        "created": 12345,
        "model": "./fake_models/ggml/fake-llama-13B.ggml.q4_2.bin",
        "choices": [{"text": str(randint(0, 9)), "index": 0, "logprobs": None, "finish_reason": None}],
    }


def load_llama(llama_cpp_model: "LlamaCppModel", cache_only_single_model: bool = True) -> LlamaCppGpu:
    try:
        model_name: str = llama_cpp_model.name
        if cache_only_single_model:
            clear_model_keys: list[str] = [key for key in LLAMA_CACHED.keys() if key != model_name]
            for cached_model_name in clear_model_keys:
                try:
                    client = LLAMA_CACHED[cached_model_name].client
                    assert isinstance(client, Llama)
                    if client.ctx is not None:
                        llama_free(client.ctx)
                    client.set_cache(None)
                    client.reset()
                    LLAMA_CACHED.pop(cached_model_name)
                except Exception as e:
                    logger.error(f"[Error] Could not free llama model: {e}")
        if model_name not in LLAMA_CACHED:
            print(f"Loading LlamaCpp model: {model_name}")
            llama_cpp = get_llama(llama_cpp_model)
            assert isinstance(llama_cpp.client, Llama)
            if llama_cpp_model.cache and llama_cpp_model.cache_size is not None:
                llama_cpp.client.set_cache(
                    LlamaCache(capacity_bytes=llama_cpp_model.cache_size),
                )
            LLAMA_CACHED[model_name] = llama_cpp
        return LLAMA_CACHED[model_name]
    except Exception as e:
        raise RuntimeError(f"[Error] Could not load llama model: {e}")


def llama_cpp_generation(
    llama_cpp_model: "LlamaCppModel",
    prompt: str,
    m_queue,  # multiprocessing.managers.AutoProxy[Queue]
    m_done,  # multiprocessing.managers.EventProxy
    user_chat_context: "UserChatContext",
    is_fake: bool = False,
    use_client_only: bool = True,
) -> None:
    try:
        try:
            m_done.clear()
            avoids: list[str] = get_stops(
                user_chat_context.user_chat_profile.user_role + ":",
            ) + get_stops(user_chat_context.user_chat_profile.ai_role + ":")
            llm = load_llama(llama_cpp_model)
            llm_client: Llama = llm.client
            llm_client.verbose = bool(llm.echo)

            retry_count: int = 0
            content_buffer: str = ""
            deleted_histories: int = 0
        except Exception as e:
            raise Exception(f"Could not initialize llama: {e}")

        while True:
            try:
                retry_count += 1
                if use_client_only:
                    generator = llm_client.create_completion(  # type: ignore
                        prompt=prompt,
                        suffix=llm.suffix,
                        max_tokens=min(
                            user_chat_context.left_tokens, user_chat_context.llm_model.value.max_tokens_per_request
                        ),
                        temperature=user_chat_context.user_chat_profile.temperature,
                        top_p=user_chat_context.user_chat_profile.top_p,
                        logprobs=llm.logprobs,
                        echo=bool(llm.echo),
                        stop=llm.stop + avoids if llm.stop is not None else avoids,
                        repeat_penalty=user_chat_context.user_chat_profile.frequency_penalty,
                        top_k=40,
                        stream=True,
                    )

                else:
                    llm.temperature = user_chat_context.user_chat_profile.temperature
                    llm.top_p = user_chat_context.user_chat_profile.top_p
                    llm.max_tokens = min(
                        user_chat_context.left_tokens, user_chat_context.llm_model.value.max_tokens_per_request
                    )
                    generator = (
                        llm.stream(prompt=prompt, stop=llm.stop + avoids if llm.stop is not None else avoids)
                        if not is_fake
                        else get_fake_response()
                    )
            except Exception as e:
                raise Exception(f"Error loading llama generator: {e}")

            try:
                if llm.echo:
                    print("[Info] Creating Response of prompt below:")
                    print(prompt, end="")
                for generation in generator:
                    if m_done.is_set() or retry_count > 10:
                        m_queue.put_nowait(Exception("Max retry count reached"))
                        m_done.set()
                        raise Exception(
                            f"Exceeded retry limit {retry_count}"
                        )  # stop generating if main process requests to stop
                    finish_reason: str | None = generation["choices"][0]["finish_reason"]  # type: ignore
                    text: str = generation["choices"][0]["text"]  # type: ignore
                    if text.replace("\u200b", "") == "":
                        continue
                    if content_buffer == "":
                        text = text.lstrip()
                        if text == "":
                            continue
                    if llm.echo:
                        print(text, end="")  # type: ignore
                    if finish_reason == "length":
                        raise ChatLengthException(
                            msg="[Warning] Incomplete model output due to max_tokens parameter or token limit"
                        )  # raise exception for token limit
                    content_buffer += text
                    m_queue.put(text)
                if content_buffer.replace("\u200b", "").strip() == "":
                    print("[Warning] Empty model output. Retrying...")
                    raise ChatContinueException(msg="Empty model output")  # raise exception for empty output
            except ChatLengthException:
                prompt += content_buffer
                deleted_histories += user_chat_context.ensure_token_not_exceed()
                deleted_histories += user_chat_context.clear_tokens(tokens_to_remove=ChatConfig.extra_token_margin)
            except ValueError as e:
                if "tokens exceed context window" in str(e):
                    prompt += content_buffer
                    deleted_histories += user_chat_context.ensure_token_not_exceed()
                    deleted_histories += user_chat_context.clear_tokens(tokens_to_remove=ChatConfig.extra_token_margin)
                    print("[Warning] Token limit exceeded. Retrying...")
                else:
                    print("[Warning] ValueError. Retrying...")
                    raise e
            except ChatBreakException:
                break
            except ChatContinueException:
                continue
            except Exception as e:
                raise Exception(f"Error while generating: {e}")
            else:
                break

        try:
            # if content_buffer starts with "user_chat_context.chat_profile.ai_role: " then remove it
            prefix_to_remove: str = f"{user_chat_context.user_chat_profile.ai_role}: "
            if content_buffer.startswith(prefix_to_remove):
                content_buffer = content_buffer[len(prefix_to_remove) :]  # noqa: E203
            m_queue.put(
                {
                    "result": {
                        "generated_text": content_buffer,
                        "n_gen_tokens": len(llm.client.tokenize(b" " + content_buffer.encode("utf-8"))),
                        "deleted_histories": deleted_histories,
                    }
                }
            )
            m_done.set()  # Notify the main process that we're done
        except Exception as e:
            raise Exception(f"Error while cleaning up: {e}")
    except Exception as e:
        logger.error("An error occurred during llama_cpp_generation.", exc_info=True)
        m_queue.put_nowait(Exception(f"[Error] {e}"))
        m_done.set()


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


if __name__ == "__main__":
    from app.dependencies import process_manager
    from app.models.chat_models import UserChatContext  # noqa: F811
    from app.models.llms import LLMModels

    m_queue = process_manager.Queue()
    m_done = process_manager.Event()
    llama_cpp_model: "LlamaCppModel" = LLMModels.wizard_vicuna_7b_uncensored.value
    llm = get_llama(llama_cpp_model)
    prompt = """"""
    llama_cpp_generation(
        llama_cpp_model=llama_cpp_model,
        prompt=prompt,
        m_queue=m_queue,
        m_done=m_done,
        user_chat_context=UserChatContext.construct_default(user_id="test_user_id", chat_room_id="test_chat_room_id"),
    )

    # if can_avoid_in_buffer(text_buffer, text, avoids):
    #     text_buffer += text
    #     for avoid in avoids:
    #         if avoid in text_buffer:
    #             remainder = text_buffer[: text_buffer.index(avoid)].strip()
    #             if remainder != "":
    #                 content_buffer += remainder
    #                 m_queue.put(remainder)
    #                 raise GptBreakException()
    #             elif content_buffer.strip() != "":
    #                 raise GptBreakException()
    #             else:
    #                 text_buffer = ""
    #                 prompt += content_buffer
    #                 raise GptContinueException()
