"""OpenAI chat wrapper."""
from __future__ import annotations

import asyncio
import inspect
from orjson import loads as orjson_loads
from orjson import JSONDecodeError
import logging
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Tuple,
    Dict,
)

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    Callbacks,
)
from langchain.chat_models.openai import (
    ChatOpenAI,
    _convert_dict_to_message,
    acompletion_with_retry,
    logger,
    _convert_message_to_dict,
)
from langchain.schema import (
    BaseMessage,
    ChatGeneration,
    ChatResult,
    LLMResult,
    RunInfo,
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
        APIConnectionError,
        APIError,
        InvalidRequestError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
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

    def _create_message_dicts(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = dict(self._invocation_params) | kwargs
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
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
                    function_arguments = orjson_loads(function_arguments_unparsed)
                except JSONDecodeError:
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

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Top Level call"""
        params = self.dict()
        params["stop"] = stop

        callback_manager = AsyncCallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        run_manager = await callback_manager.on_chat_model_start(
            {"name": self.__class__.__name__}, messages, invocation_params=params
        )

        new_arg_supported = inspect.signature(self._agenerate).parameters.get(
            "run_manager"
        )
        try:
            results = await asyncio.gather(
                *[
                    self._agenerate(m, stop=stop, run_manager=run_manager, **kwargs)
                    if new_arg_supported
                    else self._agenerate(m, stop=stop)
                    for m in messages
                ]
            )
        except (KeyboardInterrupt, Exception) as e:
            await run_manager.on_llm_error(e)
            raise e
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        await run_manager.on_llm_end(output)
        if run_manager:
            output.run = RunInfo(run_id=run_manager.run_id)
        return output

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
                    function_arguments = orjson_loads(
                        res["message"]["function_call"].get("arguments")
                    )
                except JSONDecodeError:
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
