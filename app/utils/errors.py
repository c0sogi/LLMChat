from os import getpid, kill
from re import Match, compile
from signal import SIGINT
from typing import Callable, Optional, TypedDict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from app.models.base_models import (
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
)
from app.utils.logger import ApiLogger

logger = ApiLogger(__name__)


class ErrorResponse(TypedDict):
    message: str
    type: str
    param: Optional[str]
    code: Optional[str]


class ErrorResponseCallbacks:
    @staticmethod
    def token_exceed_callback(
        request: CreateCompletionRequest | CreateChatCompletionRequest,
        match: Match[str],
    ) -> tuple[int, ErrorResponse]:
        context_window = int(match.group(2))
        prompt_tokens = int(match.group(1))
        completion_tokens = request.max_tokens
        requested_tokens = request.max_tokens + prompt_tokens
        if hasattr(request, "messages"):
            msg = (
                "This model's maximum context length is {} tokens. "
                "However, you requested {} tokens ({} in the messages, {} in the completion). "
                "Please reduce the length of the messages or completion."
            )
        else:
            msg = (
                "This model's maximum context length is {} tokens, "
                "however you requested {} tokens ({} in your prompt; {} for the completion). "
                "Please reduce your prompt; or completion length."
            )
        return 400, ErrorResponse(
            message=msg.format(
                context_window,
                requested_tokens,
                prompt_tokens,
                completion_tokens,
            ),
            type="invalid_request_error",
            param="messages",
            code="context_length_exceeded",
        )


class RouteErrorHandler(APIRoute):
    """Custom APIRoute that handles application errors and exceptions"""

    def wrap_error_message_as_openai(
        self,
        error: Exception,
        body: Optional[
            CreateCompletionRequest
            | CreateChatCompletionRequest
            | CreateEmbeddingRequest
        ] = None,
    ) -> tuple[int, ErrorResponse]:
        if body is None or isinstance(body, CreateEmbeddingRequest):
            return 500, ErrorResponse(
                message=str(error),
                type="internal_server_error",
                param=None,
                code=None,
            )
        pattern_and_callbacks = {
            compile(
                r"Requested tokens \((\d+)\) exceed context window of (\d+)"
            ): ErrorResponseCallbacks.token_exceed_callback,
        }
        for pattern, callback in pattern_and_callbacks.items():
            match = pattern.search(str(error))
            if match is not None:
                return callback(body, match)
        return 500, ErrorResponse(
            message=str(error),
            type="internal_server_error",
            param=None,
            code=None,
        )

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except (OSError, MemoryError) as e:
                logger.exception(f"Exception in llama-cpp: {e}")
                if isinstance(e, MemoryError):
                    error_msg = str(e)
                else:
                    error_msg = "Memory corruption occurred. Terminating..."
                kill(getpid(), SIGINT)
                return JSONResponse(
                    {
                        "error": {
                            "message": error_msg,
                            "type": "internal_server_error",
                            "param": None,
                            "code": None,
                        }
                    },
                    500,
                )
            except Exception as e:
                logger.exception(f"Exception in llama-cpp: {e}")

                json_body = await request.json()
                try:
                    if "messages" in json_body:
                        body = CreateChatCompletionRequest(**json_body)
                    elif "prompt" in json_body:
                        body = CreateCompletionRequest(**json_body)
                    else:
                        body = CreateEmbeddingRequest(**json_body)
                except Exception:
                    body = None
                (
                    status_code,
                    error_message,
                ) = self.wrap_error_message_as_openai(error=e, body=body)
                return JSONResponse(
                    {"error": error_message},
                    status_code=status_code,
                )

        return custom_route_handler
