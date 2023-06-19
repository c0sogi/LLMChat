from langchain import LlamaCpp
from pydantic import Field, root_validator


class CustomLlamaCpp(LlamaCpp):
    low_vram: bool = Field(False, alias="low_vram")

    @root_validator()
    def validate_environment(cls, values: dict) -> dict:
        """Validate that llama-cpp-python library is installed."""
        model_path = values["model_path"]
        model_param_names = [
            "lora_path",
            "lora_base",
            "n_ctx",
            "n_parts",
            "seed",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "n_threads",
            "n_batch",
            "use_mmap",
            "last_n_tokens_size",
        ]
        model_params = {k: values[k] for k in model_param_names}
        # For backwards compatibility, only include if non-null.
        if values["n_gpu_layers"] is not None:
            model_params["n_gpu_layers"] = values["n_gpu_layers"]
        if values["low_vram"] is not None:
            model_params["low_vram"] = values["low_vram"]

        try:
            from llama_cpp import Llama

            values["client"] = Llama(model_path, **model_params)
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import llama-cpp-python library. "
                "Please install the llama-cpp-python library to "
                "use this embedding model: pip install llama-cpp-python"
            )
        except Exception as e:
            raise ValueError(
                f"Could not load Llama model from path: {model_path}. "
                f"Received error {e}"
            )

        return values
