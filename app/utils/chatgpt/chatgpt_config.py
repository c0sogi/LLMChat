from dataclasses import dataclass, field
import re
from app.models.gpt_models import UserGptContext


@dataclass(frozen=True)
class ChatGPTConfig:
    api_url: str = "https://api.openai.com/v1/chat/completions"  # api url for openai
    wait_for_timeout: float = 60.0  # wait for this time before timeout
    wait_for_reconnect: float = 3.0  # wait for this time before reconnecting
    user_gpt_contexts: dict[str, UserGptContext] = field(default_factory=dict)
    api_regex_pattern: re.Pattern = re.compile(r"data:\s*({.+?})\n\n")


GPT_CONFIG = ChatGPTConfig()
