"""OpenAI chat wrapper."""
from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
)
from langchain.chat_models.openai import (
    ChatOpenAI,
    logger,
    acompletion_with_retry,
    _convert_dict_to_message,
)
from langchain.schema import (
    BaseMessage,
    ChatGeneration,
    ChatResult,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.errors.chat_exceptions import (
    ChatContentFilterException,
    ChatLengthException,
    ChatTooMuchTokenException,
)


def _create_retry_decorator(llm: ChatOpenAI) -> Callable[[Any], Any]:
    from openai.error import (
        Timeout,
        APIError,
        APIConnectionError,
        RateLimitError,
        ServiceUnavailableError,
        InvalidRequestError,
    )

    min_seconds = 1
    max_seconds = 60

    def retry_if_invalid_request_error(exception: BaseException):
        if isinstance(
            exception, InvalidRequestError
        ) and "maximum context length" in str(exception):
            raise ChatTooMuchTokenException(
                msg="You requested more than its maximum context length"
            )
        return False

    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(Timeout)
            | retry_if_exception_type(APIError)
            | retry_if_exception_type(APIConnectionError)
            | retry_if_exception_type(RateLimitError)
            | retry_if_exception_type(ServiceUnavailableError)
            | retry_if_exception(retry_if_invalid_request_error)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


class CustomChatOpenAI(ChatOpenAI):
    def _create_retry_decorator(self) -> Callable[[Any], Any]:
        return _create_retry_decorator(self)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        if self.streaming:
            inner_completion = ""
            role = "assistant"
            params["stream"] = True
            async for stream_resp in await acompletion_with_retry(
                self, messages=message_dicts, **params
            ):
                role = stream_resp["choices"][0]["delta"].get("role", role)
                token = stream_resp["choices"][0]["delta"].get("content", "")
                finish_reason = stream_resp["choices"][0]["finish_reason"]
                inner_completion += token
                if finish_reason == "length":
                    raise ChatLengthException(
                        msg=inner_completion
                    )  # raise exception for token limit
                elif finish_reason == "content_filter":
                    raise ChatContentFilterException(
                        msg="Omitted content due to a flag from our content filters"
                    )  # raise exception for openai content filter
                if run_manager:
                    await run_manager.on_llm_new_token(token)
            message = _convert_dict_to_message(
                {"content": inner_completion, "role": role}
            )
            return ChatResult(generations=[ChatGeneration(text="", message=message)])
        else:
            response = await acompletion_with_retry(
                self, messages=message_dicts, **params
            )
            return self._create_chat_result(response)
