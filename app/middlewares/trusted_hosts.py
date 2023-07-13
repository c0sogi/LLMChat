from typing import Sequence

from starlette.datastructures import URL, Headers
from starlette.responses import PlainTextResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.errors.api_exceptions import Responses_500


class TrustedHostMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        allowed_hosts: Sequence[str] | None = None,
        except_path: Sequence[str] | None = None,
        www_redirect: bool = True,
    ):
        self.app: ASGIApp = app
        self.allowed_hosts: list = (
            ["*"] if allowed_hosts is None else list(allowed_hosts)
        )
        self.allow_any: bool = (
            "*" in allowed_hosts if allowed_hosts is not None else True
        )
        self.www_redirect: bool = www_redirect
        self.except_path: list = (
            [] if except_path is None else list(except_path)
        )
        if allowed_hosts is not None:
            for allowed_host in allowed_hosts:
                if "*" in allowed_host[1:]:
                    raise Responses_500.middleware_exception
                if (
                    allowed_host.startswith("*") and allowed_host != "*"
                ) and not allowed_host.startswith("*."):
                    raise Responses_500.middleware_exception

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if self.allow_any or scope["type"] not in (
            "http",
            "websocket",
        ):  # pragma: no cover
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        host = headers.get("host", "").split(":")[0]
        is_valid_host = False
        found_www_redirect = False
        for pattern in self.allowed_hosts:
            if (
                host == pattern
                or pattern.startswith("*")
                and host.endswith(pattern[1:])
                or URL(scope=scope).path in self.except_path
            ):
                is_valid_host = True
                break
            elif "www." + host == pattern:
                found_www_redirect = True

        if is_valid_host:
            await self.app(scope, receive, send)
        else:
            if found_www_redirect and self.www_redirect:
                url = URL(scope=scope)
                redirect_url = url.replace(netloc="www." + url.netloc)
                response = RedirectResponse(url=str(redirect_url))
            else:
                response = PlainTextResponse(
                    "Invalid host header", status_code=400
                )
            await response(scope, receive, send)
