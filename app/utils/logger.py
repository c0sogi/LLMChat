import os
import json
import logging
from typing import Optional
from datetime import timedelta, datetime
from time import time
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response
from app.common.config import logging_config, LoggingConfig
from app.errors.api_exceptions import APIException, InternalServerError


class CustomLogger(logging.Logger):
    def __init__(self, name: str, logging_config: LoggingConfig = None) -> None:
        if logging_config is None:
            logging_config = LoggingConfig()
        super().__init__(name=name, level=logging_config.logger_level)
        formatter = logging.Formatter(logging_config.logging_format)

        console = logging.StreamHandler()
        console.setLevel(logging_config.console_log_level)
        console.setFormatter(formatter)

        if logging_config.file_log_name is not None:
            if not os.path.exists(os.path.dirname(logging_config.file_log_name)):
                os.makedirs(os.path.dirname(logging_config.file_log_name))
            file_handler = logging.FileHandler(filename=logging_config.file_log_name, mode="a", encoding="utf-8")
            file_handler.setLevel(logging_config.file_log_level)
            file_handler.setFormatter(formatter)

        self.addHandler(console)
        self.addHandler(file_handler)


class ApiLogger(CustomLogger):
    def __init__(self, name: str, logging_config: LoggingConfig = None) -> None:
        super().__init__(name=name, logging_config=logging_config)

    def _hide_email(self, email: str) -> str:
        separated_email = email.split("@")
        if len(separated_email) == 2:
            local_parts, domain = separated_email
            return "**" + local_parts[2:-1] + "*@" + domain[1]
        else:
            return "".join(separated_email)

    def _generate_error_log(
        self, error: InternalServerError | HTTPException | APIException, request: Request
    ) -> dict[str, str]:
        if request.state.inspect is not None:
            frame = request.state.inspect
            error_file = frame.f_code.co_filename
            error_func = frame.f_code.co_name
            error_line = frame.f_lineno
        else:
            error_func = error_file = error_line = "UNKNOWN"

        return {
            "errorFunc": str(error_func),
            "location": f"{error_line} line in {error_file}",
            "raised": str(error.__class__.__name__),
            "msg": str(error.ex),
        }

    def log_api(
        self,
        request: Request,
        response: Optional[Response] = None,
        error: Optional[InternalServerError | HTTPException | APIException] = None,
        **kwargs,
    ) -> None:
        processed_time = time() - request.state.start if hasattr(request.state, "start") else -1
        status_code = error.status_code if error else response.status_code
        user = request.state.user
        utc_now = datetime.utcnow()
        json_data = {
            "url": request.url.hostname + request.url.path,
            "method": str(request.method),
            "statusCode": status_code,
            "errorDetail": self._generate_error_log(error=error, request=request) if error is not None else None,
            "client": {
                "client": request.state.ip,
                "user": user.id if user and user.id else None,
                "email": self._hide_email(email=user.email) if user and user.email else None,
            },
            "processedTime": str(round(processed_time * 1000, 5)) + "ms",
            "datetimeUTC": utc_now.strftime("%Y/%m/%d %H:%M:%S"),
            "datetimeKST": (utc_now + timedelta(hours=9)).strftime("%Y/%m/%d %H:%M:%S"),
        } | kwargs
        log = json.dumps(json_data)
        self.error(log, exc_info=True) if error and error.status_code >= 500 else self.info(log)


api_logger = ApiLogger("FastAPI", logging_config=logging_config)

if __name__ == "__main__":
    api_logger.debug("Testing logger: debug")
    api_logger.info("Testing logger: info")
    api_logger.warning("Testing logger: warning")
    try:
        raise Exception("Test exception")
    except Exception:
        api_logger.error("Testing logger: error", exc_info=True)
    api_logger.critical("Testing logger: critical")
