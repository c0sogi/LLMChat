import sys
from pathlib import Path

from torch import cuda

assert cuda.is_available()
from pathlib import Path
from typing import Any

from app.utils.logger import ApiLogger

sys.path.insert(0, str(Path("repositories/exllama")))
from repositories.exllama.generator import ExLlamaGenerator
from repositories.exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from repositories.exllama.tokenizer import ExLlamaTokenizer

logger = ApiLogger("||exllama||")


class ExllamaModel:
    config: ExLlamaConfig
    model: ExLlama
    cache: ExLlamaCache
    tokenizer: ExLlamaTokenizer

    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path_to_model: str | Path):
        path_to_model = Path("models") / Path(path_to_model)
        tokenizer_model_path = path_to_model / "tokenizer.model"
        model_config_path = path_to_model / "config.json"

        # Find the model checkpoint
        model_path = None
        for ext in [".safetensors", ".pt", ".bin"]:
            found = list(path_to_model.glob(f"*{ext}"))
            if len(found) > 0:
                if len(found) > 1:
                    logger.warning(
                        f"More than one {ext} model has been found. The last one will be selected. It could be wrong."
                    )

                model_path = found[-1]
                break
        config = ExLlamaConfig(str(model_config_path))
        config.model_path = str(model_path)  # type: ignore
        model = ExLlama(config)
        tokenizer = ExLlamaTokenizer(str(tokenizer_model_path))
        cache = ExLlamaCache(model)

        result = cls()
        result.config = config
        result.model = model
        result.cache = cache
        result.tokenizer = tokenizer
        return result, result

    def generate(self, prompt: str, state: dict, callback: Any = None):
        generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        generator.settings.temperature = state["temperature"]
        generator.settings.top_p = state["top_p"]
        generator.settings.top_k = state["top_k"]
        generator.settings.typical = state["typical_p"]
        generator.settings.token_repetition_penalty_max = state["repetition_penalty"]
        if state["ban_eos_token"]:
            generator.disallow_tokens([self.tokenizer.eos_token_id])

        text = generator.generate_simple(prompt, max_new_tokens=state["max_new_tokens"])
        return text

    def generate_with_streaming(self, prompt: str, state: dict, callback: Any = None):
        generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        generator.settings.temperature = state["temperature"]
        generator.settings.top_p = state["top_p"]
        generator.settings.top_k = state["top_k"]
        generator.settings.typical = state["typical_p"]
        generator.settings.token_repetition_penalty_max = state["repetition_penalty"]
        if state["ban_eos_token"]:
            generator.disallow_tokens([self.tokenizer.eos_token_id])

        generator.end_beam_search()
        ids = generator.tokenizer.encode(prompt)
        generator.gen_begin(ids)
        initial_len = generator.sequence[0].shape[0]
        for _ in range(state["max_new_tokens"]):
            token = generator.gen_single_token()
            yield (generator.tokenizer.decode(generator.sequence[0][initial_len:]))
            if token.item() == generator.tokenizer.eos_token_id:
                break

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string)
