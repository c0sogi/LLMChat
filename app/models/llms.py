from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from langchain import PromptTemplate

from app.common.config import OPENAI_API_KEY, ChatConfig
from app.common.constants import ChatTurnTemplates, DescriptionTemplates
from app.mixins.enum import EnumMixin

from .base_models import UserChatRoles
from .llm_tokenizers import (
    BaseTokenizer,
    ExllamaTokenizer,
    LlamaTokenizer,
    OpenAITokenizer,
)


@dataclass
class LLMModel:
    name: str = "llm_model"  # model name
    max_total_tokens: int = 2048
    max_tokens_per_request: int = 1024
    tokenizer: BaseTokenizer = field(
        default_factory=lambda: OpenAITokenizer(model_name="gpt-3.5-turbo")
    )
    user_chat_roles: UserChatRoles = field(
        default_factory=lambda: UserChatRoles(
            ai="assistant",
            system="system",
            user="user",
        ),
    )
    prefix_template: Optional[
        Union[PromptTemplate, str]
    ] = None  # A prefix to prepend to the generated text. If None, no prefix is prepended.
    suffix_template: Optional[
        Union[PromptTemplate, str]
    ] = None  # A suffix to prepend to the generated text. If None, no suffix is prepended.
    token_margin: int = field(
        default=8,
        metadata={
            "description": " The marginal number of tokens when counting the number of tokens in context."
            "The more token_margin, the more conservative the model will be when counting context tokens."
        },
    )
    prefix: Optional[str] = field(init=False, repr=False, default=None)
    suffix: Optional[str] = field(init=False, repr=False, default=None)

    @staticmethod
    def _prepare_format(
        input_variables: list[str],
        predefined_format: dict[str, str],
    ) -> dict[str, str | None]:
        return dict(
            zip(
                input_variables,
                map(
                    predefined_format.get,
                    input_variables,
                ),
            )
        )

    def __post_init__(self):
        user_chat_roles = self.user_chat_roles
        predefined_format = {
            "user": user_chat_roles.user,
            "USER": user_chat_roles.user,
            "ai": user_chat_roles.ai,
            "AI": user_chat_roles.ai,
            "char": user_chat_roles.ai,
            "system": user_chat_roles.system,
            "SYSTEM": user_chat_roles.system,
        }
        # If the global prefix is None, then use the prefix template
        if ChatConfig.global_prefix is None:
            if isinstance(self.prefix_template, PromptTemplate):
                # format the template with the predefined format, only for input variables
                self.prefix = self.prefix_template.format(
                    **self._prepare_format(
                        self.prefix_template.input_variables, predefined_format
                    )
                )
            elif isinstance(self.prefix_template, str):
                self.prefix = self.prefix_template.format(**predefined_format)
            else:
                self.prefix = None
        else:
            self.prefix = ChatConfig.global_prefix

        # If the global suffix is None, then use the suffix template
        if ChatConfig.global_suffix is None:
            if isinstance(self.suffix_template, PromptTemplate):
                # format the template with the predefined format, only for input variables
                self.suffix = self.suffix_template.format(
                    **self._prepare_format(
                        self.suffix_template.input_variables, predefined_format
                    )
                )
            elif isinstance(self.suffix_template, str):
                self.suffix = self.suffix_template.format(**predefined_format)
            else:
                self.suffix = None
        else:
            self.suffix = ChatConfig.global_suffix
        self._prefix_tokens: Optional[int] = None
        self._suffix_tokens: Optional[int] = None

    @property
    def prefix_tokens(self) -> int:
        # Lazy load the prefix tokens
        if self.prefix is None:
            return 0
        if self._prefix_tokens is None:
            self._prefix_tokens = (
                self.tokenizer.tokens_of(self.prefix) + self.token_margin
            )
        return self._prefix_tokens

    @property
    def suffix_tokens(self) -> int:
        # Lazy load the suffix tokens
        if self.suffix is None:
            return 0
        if self._suffix_tokens is None:
            self._suffix_tokens = (
                self.tokenizer.tokens_of(self.suffix) + self.token_margin
            )
        return self._suffix_tokens


@dataclass
class LlamaCppModel(LLMModel):
    """Llama.cpp model that can be loaded from local path."""

    tokenizer: LlamaTokenizer = field(
        default_factory=lambda: LlamaTokenizer(model_name=""),
        metadata={"description": "The tokenizer to use for this model."},
    )
    name: str = field(default="Llama C++")
    model_path: str = field(
        default="YOUR_GGML.bin"
    )  # The path to the model. Must end with .bin. You must put .bin file into "llama_models/ggml"
    user_chat_roles: UserChatRoles = field(
        default_factory=lambda: UserChatRoles(
            ai="ASSISTANT",
            system="SYSTEM",
            user="USER",
        ),
    )
    prefix_template: Optional[Union[PromptTemplate, str]] = field(
        default_factory=lambda: DescriptionTemplates.USER_AI__DEFAULT,
    )
    chat_turn_prompt: PromptTemplate = field(
        default_factory=lambda: ChatTurnTemplates.ROLE_CONTENT_1
    )  # The prompt to use for chat turns.
    n_parts: int = (
        -1
    )  # Number of parts to split the model into. If -1, the number of parts is automatically determined.
    n_gpu_layers: int = 30  # Number of layers to keep on the GPU. If 0, all layers are kept on the GPU.
    seed: int = -1  # Seed. If -1, a random seed is used.
    f16_kv: bool = True  # Use half-precision for key/value cache.
    logits_all: bool = (
        False  # Return logits for all tokens, not just the last token.
    )
    vocab_only: bool = False  # Only load the vocabulary, no weights.
    use_mlock: bool = True  # Force system to keep model in RAM.
    n_batch: int = 512  # Number of tokens to process in parallel. Should be a number between 1 and n_ctx.
    last_n_tokens_size: int = 64  # The number of tokens to look back when applying the repeat_penalty.
    use_mmap: bool = True  # Whether to use memory mapping for the model.
    streaming: bool = True  # Whether to stream the results, token by token.
    cache: bool = (
        False  # The size of the cache in bytes. Only used if cache is True.
    )
    echo: bool = True  # Whether to echo the prompt.
    lora_base: Optional[str] = None  # The path to the Llama LoRA base model.
    lora_path: Optional[
        str
    ] = None  # The path to the Llama LoRA. If None, no LoRa is loaded.
    cache_type: Optional[Literal["disk", "ram"]] = "ram"
    cache_size: Optional[int] = (
        2 << 30
    )  # The size of the cache in bytes. Only used if cache is True.
    n_threads: Optional[
        int
    ] = None  # Number of threads to use. If None, the number of threads is automatically determined.
    low_vram: bool = False  # Whether to use less VRAM.
    embedding: bool = False  # Whether to use the embedding layer.
    rope_freq_base: float = 10000.0  # I use 26000 for n_ctx=4096. https://github.com/ggerganov/llama.cpp/pull/2054
    rope_freq_scale: float = 1.0  # Generally, 2048 / n_ctx. https://github.com/ggerganov/llama.cpp/pull/2054


@dataclass
class ExllamaModel(LLMModel):
    """Exllama model that can be loaded from local path."""

    model_path: str = field(
        default="YOUR_GPTQ_FOLDER_NAME",
        metadata={
            "description": "The GPTQ model path to the model."
            "e.g. If you have a model folder in 'llama_models/gptq/your_model',"
            "then you should set this to 'your_model'."
        },
    )
    tokenizer: ExllamaTokenizer = field(
        default_factory=lambda: ExllamaTokenizer(model_name=""),
        metadata={"description": "The tokenizer to use for this model."},
    )

    user_chat_roles: UserChatRoles = field(
        default_factory=lambda: UserChatRoles(
            ai="ASSISTANT",
            system="SYSTEM",
            user="USER",
        ),
    )
    prefix_template: Optional[Union[PromptTemplate, str]] = field(
        default_factory=lambda: DescriptionTemplates.USER_AI__DEFAULT,
    )
    chat_turn_prompt: PromptTemplate = field(
        default_factory=lambda: ChatTurnTemplates.ROLE_CONTENT_1,
        metadata={"description": "The prompt to use for each chat turn."},
    )
    compress_pos_emb: float = field(
        default=1.0,
        metadata={
            "description": "Increase to compress positional embeddings applied to sequence."
            "This is useful when you want to extend context window size."
            "e.g. If you want to extend context window size from 2048 to 4096, set this to 2.0."
        },
    )
    gpu_peer_fix: bool = field(
        default=False,
        metadata={
            "description": "Apparently Torch can have problems transferring tensors directly 1 GPU to another."
            "Enable this to use system RAM as a buffer for GPU to GPU transfers."
        },
    )
    auto_map: Optional[list[float]] = field(
        default=None,
        metadata={
            "description": "List of floats with memory allocation in GB, per CUDA device, overrides device_map"
        },
    )
    # Optional parameters
    matmul_recons_thd: int = 8
    fused_mlp_thd: int = 2
    sdp_thd: int = 8
    fused_attn: bool = True
    matmul_fused_remap: bool = False
    rmsnorm_no_half2: bool = False
    rope_no_half2: bool = False
    matmul_no_half2: bool = False
    silu_no_half2: bool = False
    concurrent_streams: bool = False


@dataclass
class OpenAIModel(LLMModel):
    api_url: str = "https://api.openai.com/v1/chat/completions"
    api_key: str | None = field(repr=False, default=None)
    user_chat_roles: UserChatRoles = field(
        default_factory=lambda: UserChatRoles(
            ai="assistant",
            system="system",
            user="user",
        ),
    )


class LLMModels(EnumMixin):
    #  OpenAI models
    gpt_3_5_turbo = OpenAIModel(
        name="gpt-3.5-turbo",
        max_total_tokens=4096,
        max_tokens_per_request=2048,
        token_margin=8,
        tokenizer=OpenAITokenizer("gpt-3.5-turbo"),
        api_url="https://api.openai.com/v1/chat/completions",
        api_key=OPENAI_API_KEY,
        # prefix_template=PromptTemplate(
        #     template="You'll be roleplaying with the user, so respond to their comments as if they're annoying you.",
        #     input_variables=[],
        # ),  # Example of a prefix template
        # suffix_template=PromptTemplate(
        #     template="You must respond to the user in Korean.",
        #     input_variables=[],
        # ),  # Example of a suffix template
    )
    gpt_3_5_turbo_16k = OpenAIModel(
        name="gpt-3.5-turbo-16k",
        max_total_tokens=16384,
        max_tokens_per_request=8192,
        token_margin=8,
        tokenizer=OpenAITokenizer("gpt-3.5-turbo"),
        api_url="https://api.openai.com/v1/chat/completions",
        api_key=OPENAI_API_KEY,
    )

    gpt_4 = OpenAIModel(
        name="gpt-4",
        max_total_tokens=8192,
        max_tokens_per_request=4096,
        token_margin=8,
        tokenizer=OpenAITokenizer("gpt-4"),
        api_url="https://api.openai.com/v1/chat/completions",
        api_key=OPENAI_API_KEY,
    )

    # Llama-cpp models
    wizard_vicuna_13b_uncensored = LlamaCppModel(
        name="Wizard-Vicuna-13B-Uncensored",
        max_total_tokens=4096,  # context tokens (n_ctx)
        max_tokens_per_request=2048,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("ehartford/Wizard-Vicuna-13B-Uncensored"),
        model_path="Wizard-Vicuna-13B-Uncensored.ggmlv3.q5_1.bin",  # The filename of model. Must end with .bin.
        prefix_template=DescriptionTemplates.USER_AI__DEFAULT,
    )
    gorilla_7b = LlamaCppModel(
        name="gorilla-7B-GGML",
        max_total_tokens=2048,  # context tokens (n_ctx)
        max_tokens_per_request=1024,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("gorilla-llm/gorilla-7b-hf-delta-v0"),
        model_path="Gorilla-7B.ggmlv3.q3_K_S.bin",  # The filename of model. Must end with .bin.
        prefix_template=DescriptionTemplates.USER_AI__DEFAULT,
    )
    manticore_13b_uncensored = LlamaCppModel(
        name="Manticore-13B-GGML",
        max_total_tokens=2048,  # context tokens (n_ctx)
        max_tokens_per_request=1024,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("openaccess-ai-collective/manticore-13b"),
        model_path="Manticore-13B.ggmlv2.q5_1.bin",  # The filename of model. Must end with .bin.
    )
    kovicuna_7b = LlamaCppModel(
        name="kovicuna_7b",
        max_total_tokens=2048,  # context tokens (n_ctx)
        max_tokens_per_request=1024,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("digitous/13B-HyperMantis"),
        model_path="kovicuna_q4km.bin",  # The filename of model. Must end with .bin.
        prefix_template=DescriptionTemplates.USER_AI__SHORT,
    )
    wizard_lm_13b_v1_1 = LlamaCppModel(
        name="wizardLM-13B-Uncensored",
        max_total_tokens=4096,  # context tokens (n_ctx)
        max_tokens_per_request=2048,  # The maximum number of tokens to generate.
        token_margin=8,
        prefix_template=DescriptionTemplates.USER_AI__SHORT,
        tokenizer=LlamaTokenizer("victor123/WizardLM-13B-1.0"),
        model_path="wizardlm-13b-v1.1.ggmlv3.q4_K_S.bin",  # The filename of model. Must end with .bin.
        rope_freq_scale=0.5,  # similar to `compress_pos_emb`, but inverse number: (2048 / n_ctx)
        rope_freq_base=26000,  # need some perplexity test; 26000 is my empirical value for 4096 context tokens
    )
    guanaco_13b = LlamaCppModel(
        name="guanaco-13B-GGML",
        max_total_tokens=2048,  # context tokens (n_ctx)
        max_tokens_per_request=1024,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer(
            "timdettmers/guanaco-65b-merged"
        ),  # timdettmers/guanaco-13b
        model_path="guanaco-13B.ggmlv3.q5_1.bin",  # The filename of model. Must end with .bin.
        prefix_template=DescriptionTemplates.USER_AI__SHORT,
        user_chat_roles=UserChatRoles(
            user="Human",
            ai="Assistant",
            system="Instruction",
        ),
    )
    karen_the_editor_13b = LlamaCppModel(
        name="Karen_theEditor_13B-GGML",
        max_total_tokens=2048,  # context tokens (n_ctx)
        max_tokens_per_request=1024,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("FPHam/Karen_theEditor_13b_HF"),
        model_path="Karen-The-Editor.ggmlv3.q5_1.bin",  # The filename of model. Must end with .bin.
        prefix_template=DescriptionTemplates.USER_AI__SHORT,
        user_chat_roles=UserChatRoles(
            user="USER",
            ai="ASSISTANT",
            system="SYSTEM",
        ),
    )
    airoboros_13b = LlamaCppModel(
        name="airoboros-13b-gpt4-GGML",
        max_total_tokens=4096,  # context tokens (n_ctx)
        max_tokens_per_request=2048,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("jondurbin/airoboros-13b-gpt4"),
        model_path="airoboros-13b-gpt4.ggmlv3.q5_1.bin",  # The filename of model. Must end with .bin.
        prefix_template=DescriptionTemplates.USER_AI__SHORT,
    )
    selfee_7b = LlamaCppModel(
        name="selfee-7B-GGML",
        max_total_tokens=4096,  # context tokens (n_ctx)
        max_tokens_per_request=2048,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("kaist-ai/selfee-7b-delta"),
        model_path="selfee-7b-superhot-8k.ggmlv3.q4_1.bin",  # The filename of model. Must end with .bin.
        prefix_template=DescriptionTemplates.USER_AI__SHORT,
        rope_freq_scale=0.5,  # similar to `compress_pos_emb`, but inverse number: (2048 / n_ctx)
        rope_freq_base=26000,  # need some perplexity test; 26000 is my empirical value for 4096 context tokens
    )
    llama_7b = LlamaCppModel(
        name="llama-7b-GGML",
        max_total_tokens=2048,  # context tokens (n_ctx)
        max_tokens_per_request=1024,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("HuggingFaceM4/llama-7b-tokenizer"),
        model_path="llama-7b.ggmlv3.q5_K_M.bin",  # The filename of model. Must end with .bin.
        prefix_template=None,
        embedding=True,
    )
    orca_mini_3b = LlamaCppModel(
        name="orca_mini_3B-GGML",
        max_total_tokens=2048,  # context tokens (n_ctx)
        max_tokens_per_request=1024,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("psmathur/orca_mini_3b"),
        model_path="orca-mini-3b.ggmlv3.q4_1.bin",  # The filename of model. Must end with .bin.
        chat_turn_prompt=ChatTurnTemplates.ROLE_CONTENT_2,
        user_chat_roles=UserChatRoles(
            user="User",
            ai="Response",
            system="System",
        ),
    )
    airoboros_33b = LlamaCppModel(
        name="airoboros-33b-gpt4-1.4-GGML",
        max_total_tokens=5120,  # context tokens (n_ctx)
        max_tokens_per_request=2560,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("jondurbin/airoboros-33b-gpt4-1.4"),
        model_path="airoboros-33b-gpt4-1.4.ggmlv3.q3_K_S.bin",  # The filename of model. Must end with .bin.
        n_gpu_layers=26,
        chat_turn_prompt=ChatTurnTemplates.ROLE_CONTENT_5,
        user_chat_roles=UserChatRoles(
            user="USER",
            ai="ASSISTANT",
            system="SYSTEM",
        ),
    )
    chronos_hermes_13b = LlamaCppModel(
        name="TheBloke/chronos-hermes-13B-GGML",
        max_total_tokens=4096,  # context tokens (n_ctx)
        max_tokens_per_request=2048,  # The maximum number of tokens to generate.
        token_margin=8,
        tokenizer=LlamaTokenizer("Austism/chronos-hermes-13b"),
        model_path="chronos-hermes-13b.ggmlv3.q4_K_M.bin",  # The filename of model. Must end with .bin.
        chat_turn_prompt=ChatTurnTemplates.ROLE_CONTENT_1,
        user_chat_roles=UserChatRoles(
            user="User",
            ai="Assistant",
            system="System",
        ),
    )
    orca_mini_7b = ExllamaModel(
        model_path="orca_mini_7b",
        name="orca_mini_7b",
        max_total_tokens=2048,
        max_tokens_per_request=2048,
        token_margin=8,
        tokenizer=ExllamaTokenizer("orca_mini_7b"),
        compress_pos_emb=2.0,
        prefix_template=DescriptionTemplates.USER_AI__GAME,
        user_chat_roles=UserChatRoles(
            user="Player",
            ai="Narrator",
            system="System",
        ),
    )
    longchat_7b = ExllamaModel(
        model_path="longchat_7b",
        name="longchat_7b",
        max_total_tokens=16384,
        max_tokens_per_request=8192,
        token_margin=8,
        tokenizer=ExllamaTokenizer("longchat_7b"),
        compress_pos_emb=8.0,
        prefix_template=DescriptionTemplates.USER_AI__GAME,
        user_chat_roles=UserChatRoles(
            user="Player",
            ai="Narrator",
            system="System",
        ),
    )

    @classmethod
    def find_model_by_name(cls, name: str) -> LLMModel | None:
        for model in cls:
            if model.value.name == name or model.name == name:
                return model.value
        return None
