"""OpenAI chat wrapper."""
from __future__ import annotations
import json

import logging
from typing import (
    Any,
    Callable,
    List,
    Mapping,
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
            completion_tokens = 0
            inner_completion = ""
            function_name = ""
            function_arguments_unparsed = ""
            role = "assistant"
            params["stream"] = True
            async for stream_resp in await acompletion_with_retry(
                self, messages=message_dicts, **params
            ):
                role = stream_resp["choices"][0]["delta"].get("role", role)
                token = stream_resp["choices"][0]["delta"].get("content", "")
                function_call = stream_resp["choices"][0]["delta"].get(
                    "function_call", None
                )
                finish_reason = stream_resp["choices"][0]["finish_reason"]
                if token:
                    inner_completion += token
                    completion_tokens += 1
                    if run_manager:
                        await run_manager.on_llm_new_token(token)
                if function_call is not None:
                    function_name += function_call.get("name", "")
                    function_arguments_unparsed += function_call.get("arguments", "")
                if finish_reason == "length":
                    raise ChatLengthException(
                        msg=inner_completion
                    )  # raise exception for token limit
                elif finish_reason == "content_filter":
                    raise ChatContentFilterException(
                        msg="Omitted content due to a flag from our content filters"
                    )  # raise exception for openai content filter
            message = _convert_dict_to_message(
                {"content": inner_completion, "role": role}
            )
            llm_output = {
                "model_name": self.model_name,
                "token_usage": {"completion_tokens": completion_tokens},
            }
            if function_name:
                function_arguments = {}
                try:
                    function_arguments = json.loads(function_arguments_unparsed)
                except json.JSONDecodeError:
                    pass
                finally:
                    llm_output.update(
                        {
                            "function_calls": [
                                {
                                    "name": function_name,
                                    "arguments": function_arguments,
                                }
                            ]
                        }
                    )
            return ChatResult(
                generations=[ChatGeneration(text="", message=message)],
                llm_output=llm_output,
            )
        else:
            response = await acompletion_with_retry(
                self, messages=message_dicts, **params
            )
            return self._create_chat_result(response)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        overall_function_call: list[dict] = []
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
            if "function_calls" in output:
                overall_function_call.extend(output["function_calls"])
        return {
            "token_usage": overall_token_usage,
            "model_name": self.model_name,
            "function_calls": overall_function_call,
        }

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        function_calls = []
        for res in response["choices"]:
            gen = ChatGeneration(
                text="",
                message=_convert_dict_to_message(
                    {
                        "role": res["message"]["role"],
                        "content": ""
                        if res["message"]["content"] is None
                        else res["message"]["content"],
                    }
                ),
            )
            generations.append(gen)
            if res["message"].get("function_call") is not None:
                function_arguments = {}
                try:
                    function_arguments = json.loads(
                        res["message"]["function_call"].get("arguments")
                    )
                except json.JSONDecodeError:
                    pass
                finally:
                    function_calls.append(
                        {
                            "name": res["message"]["function_call"]["name"],
                            "arguments": function_arguments,
                        }
                    )
        llm_output = {
            "token_usage": response["usage"],
            "model_name": self.model_name,
            "function_calls": function_calls,
        }
        return ChatResult(
            generations=generations,
            llm_output=llm_output,
        )
