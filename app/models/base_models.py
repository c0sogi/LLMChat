from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional, Union
from uuid import uuid4

from pydantic import Field
from pydantic.main import BaseModel

from app.utils.date_utils import UTC
from app.viewmodels.status import UserStatus

from .function_calling.base import FunctionProperty


class UserRegister(BaseModel):
    email: str
    password: str


class SnsType(str, Enum):
    EMAIL = "email"
    FACEBOOK = "facebook"
    GOOGLE = "google"
    KAKAO = "kakao"


class Token(BaseModel):
    Authorization: str


class EmailRecipients(BaseModel):
    name: str
    email: str


class SendEmail(BaseModel):
    email_to: list[EmailRecipients]


class KakaoMsgBody(BaseModel):
    msg: str


class MessageOk(BaseModel):
    message: str = Field(default="OK")


class UserToken(BaseModel):
    id: int
    status: UserStatus
    email: Optional[str] = None
    name: Optional[str] = None

    class Config:
        orm_mode = True


class UserMe(BaseModel):
    id: int
    email: Optional[str] = None
    name: Optional[str] = None
    phone_number: Optional[str] = None
    profile_img: Optional[str] = None
    sns_type: Optional[str] = None

    class Config:
        orm_mode = True


class AddApiKey(BaseModel):
    user_memo: Optional[str] = None

    class Config:
        orm_mode = True


class GetApiKey(AddApiKey):
    id: int
    access_key: str
    created_at: datetime


class GetApiKeyFirstTime(GetApiKey):
    secret_key: str


class CreateApiWhiteList(BaseModel):
    ip_address: str


class GetApiWhiteList(CreateApiWhiteList):
    id: int

    class Config:
        orm_mode = True


class CreateChatMessage(BaseModel):  # stub
    message: str
    role: str
    user_id: int
    chat_room_id: str

    class Config:
        orm_mode = True


class MessageToWebsocket(BaseModel):
    msg: Optional[str]
    finish: bool
    chat_room_id: Optional[str] = None
    actual_role: Optional[str] = None
    init: bool = False
    model_name: Optional[str] = None
    uuid: Optional[str] = None
    wait_next_query: Optional[bool] = None

    class Config:
        orm_mode = True


class MessageFromWebsocket(BaseModel):
    msg: str
    translate: Optional[str] = None
    chat_room_id: str


class CreateChatRoom(BaseModel):  # stub
    chat_room_type: str
    name: str
    description: Optional[str] = None
    user_id: int

    class Config:
        orm_mode = True


class APIChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"] = Field(
        default="user",
        description="The role of the messages author. One of system, user, assistant, or function",
    )
    content: str = Field(
        default="",
        description=(
            "The contents of the message. content is required for all messages, "
            "and may be null for assistant messages with function calls."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "The name of the author of this message. name is required if role is function, "
            "and it should be the name of the function whose response is in the content. "
            "May contain a-z, A-Z, 0-9, and underscores, with a maximum length of 64 characters."
        ),
    )
    function_call: Optional[FunctionProperty] = Field(
        default=None,
        description="The name and arguments of a function that should be called, as generated by the model.",
    )

    class Config:
        orm_mode = True


class MessageHistory(BaseModel):
    role: str
    content: str
    tokens: int = 0
    timestamp: int = Field(default_factory=UTC.timestamp)
    uuid: str = Field(default_factory=lambda: uuid4().hex)
    actual_role: Optional[str] = None
    model_name: Optional[str] = None
    summarized: Optional[str] = None
    summarized_tokens: Optional[int] = None

    class Config:
        orm_mode = True

    def __repr__(self) -> str:
        if self.summarized is not None:
            return (
                f'<{self.role} uuid="{self.uuid}" date="{self.datetime}Z" tokens="{self.tokens}" '
                f'summarized="{self.summarized}">{self.content}</>'
            )
        return f'<{self.role} uuid="{self.uuid}" date="{self.datetime}Z" tokens="{self.tokens}">{self.content}</>'

    @property
    def datetime(self) -> datetime:
        try:
            return UTC.timestamp_to_datetime(self.timestamp)
        except Exception:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    @property
    def is_prefix(self) -> bool:
        return self.timestamp < 0

    @property
    def is_suffix(self) -> bool:
        return self.timestamp >= 2**50


class InitMessage(BaseModel):
    previous_chats: Optional[list[dict]] = None
    chat_rooms: Optional[list[dict[str, str]]] = None
    models: Optional[list[str]] = None
    selected_model: Optional[str] = None
    tokens: Optional[int] = None


class StreamProgress(BaseModel):
    response: str = ""
    buffer: str = ""
    uuid: Optional[str] = None


class UserChatRoles(BaseModel):
    ai: str
    system: str
    user: str


class SummarizedResult(BaseModel):
    user_id: str
    chat_room_id: str
    role: str
    content: str
    uuid: str


class ParserDefinitions(BaseModel):
    selector: Optional[str] = None
    render_js: bool = False


class TextGenerationSettings(BaseModel):
    completion_id: str = Field(
        default_factory=lambda: f"cmpl-{str(uuid4())}",
        description="The unique ID of the text generation",
    )
    max_tokens: int = Field(
        default=128,
        ge=1,
        description="The maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description=(
            "Adjust the randomness of the generated text."
            "Temperature is a hyperparameter that controls the randomness of the generated te"
            "xt. It affects the probability distribution of the model's output tokens. A high"
            "er temperature (e.g., 1.5) makes the output more random and creative, while a lo"
            "wer temperature (e.g., 0.5) makes the output more focused, deterministic, and co"
            "nservative. The default value is 0.8, which provides a balance between randomnes"
            "s and determinism. At the extreme, a temperature of 0 will always pick the most "
            "likely next token, leading to identical outputs in each run."
        ),
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Limit the next token selection to a subset of tokens with a cumulative probabili"
            "ty above a threshold P. Top-p sampling, also known as nucleus sampling, "
            "is another text generation method that selects the next token from a subset of t"
            "okens that together have a cumulative probability of at least p. This method pro"
            "vides a balance between diversity and quality by considering both the probabilit"
            "ies of tokens and the number of tokens to sample from. A higher value for top_p "
            "(e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) wil"
            "l generate more focused and conservative text."
        ),
    )
    typical_p: float = Field(
        default=0.0,
        description="Locally typical sampling threshold, 0.0 to disable typical sampling",
    )
    logprobs: Optional[int] = Field(
        default=None,
        description="The number of logprobs to return. If None, no logprobs are returned.",
    )
    echo: bool = Field(
        default=False,
        description="If True, the input is echoed back in the output.",
    )
    stop: Optional[str | list[str]] = Field(
        default=None,
        description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Positive values penalize new tokens based on their existing frequency in the tex"
            "t so far, decreasing the model's likelihood to repeat the same line verbatim."
        ),
    )

    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Positive values penalize new tokens based on whether they appear in the text so far, increasing "
            "the model's likelihood to talk about new topics."
        ),
    )
    repeat_penalty: float = Field(
        default=1.1,
        ge=0.0,
        description=(
            "A penalty applied to each token that is already generated. This helps prevent th"
            "e model from repeating itself. Repeat penalty is a hyperparameter used t"
            "o penalize the repetition of token sequences during text generation. It helps pr"
            "event the model from generating repetitive or monotonous text. A higher value (e"
            ".g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0."
            "9) will be more lenient."
        ),
    )
    top_k: int = Field(
        default=40,
        ge=0,
        description=(
            "Limit the next token selection to the K most probable tokens. Top-k samp"
            "ling is a text generation method that selects the next token only from the top k"
            " most likely tokens predicted by the model. It helps reduce the risk of generati"
            "ng low-probability or nonsensical tokens, but it may also limit the diversity of"
            " the output. A higher value for top_k (e.g., 100) will consider more tokens and "
            "lead to more diverse text, while a lower value (e.g., 10) will focus on the most"
            " probable tokens and generate more conservative text."
        ),
    )
    tfs_z: float = Field(
        default=1.0,
        description="Modify probability distribution to carefully cut off least likely tokens",
    )
    mirostat_mode: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Enable Mirostat constant-perplexity algorithm of the specified version (1 or 2; 0 = disabled)",
    )
    mirostat_tau: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description=(
            "Mirostat target entropy, i.e. the target perplexity - lower values produce focused and coherent text, "
            "larger values produce more diverse and less coherent text"
        ),
    )
    mirostat_eta: float = Field(
        default=0.1, ge=0.001, le=1.0, description="Mirostat learning rate"
    )
    logit_bias: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "A dictionary of logit bias values to use for each token. The keys are the tokens"
            " and the values are the bias values. The bias values are added to the logits of "
            "the model to influence the next token probabilities. For example, a bias value o"
            "f 5.0 will make the model 10 times more likely to select that token than it woul"
            "d be otherwise. A bias value of -5.0 will make the model 10 times less likely to"
            " select that token than it would be otherwise. The bias values are added to the "
            "logits of the model to influence."
        ),
    )
    logit_bias_type: Literal["input_ids", "tokens"] = Field(
        default="tokens",
        description=(
            "The type of logit bias to use. If 'input_ids', the bias is applied to the input"
            " ids(integer). If 'tokens', the bias is applied to the tokens(string). If None, the bias is not "
            "applied."
        ),
    )
    ban_eos_token: bool = Field(
        default=False,
        description="If True, the EOS token is banned from being generated.",
    )


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


class CreateCompletionRequest(TextGenerationSettings):
    model: str = Field(
        default=..., description="The model to use for completion."
    )
    prompt: str = Field(
        default="", description="The prompt to use for completion."
    )
    stream: bool = Field(
        default=False, description="Whether to stream the response."
    )

    class Config:
        schema_extra = {
            "example": {
                "model": "llama_7b",
                "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                "stop": ["\n", "###"],
            }
        }


class CreateChatCompletionRequest(TextGenerationSettings):
    model: str = Field(
        default=..., description="The model to use for completion."
    )
    messages: list[APIChatMessage] = Field(
        default=[],
        description="A list of messages to generate completions for.",
    )
    stream: bool = Field(
        default=False, description="Whether to stream the response."
    )

    class Config:
        schema_extra = {
            "example": {
                "model": "llama_7b",
                "messages": [
                    APIChatMessage(
                        role="system", content="You are a helpful assistant."
                    ),
                    APIChatMessage(
                        role="user", content="What is the capital of France?"
                    ),
                ],
            }
        }
