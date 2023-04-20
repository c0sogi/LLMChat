import json
import logging
from typing import Optional
from datetime import timedelta, datetime
from time import time
from fastapi import HTTPException
from fastapi.logger import logger
from starlette.requests import Request
from starlette.responses import Response

from app.errors.api_exceptions import APIException, InternalServerError


logger.setLevel(logging.INFO)


def hide_email(email: str) -> str:
    separated_email = email.split("@")
    if len(separated_email) == 2:
        local_parts, domain = separated_email
        return "**" + local_parts[2:-1] + "*@" + domain[1]
    else:
        return "".join(separated_email)


def error_log_generator(
    error: InternalServerError | HTTPException | APIException, request: Request
) -> dict[str, any]:
    if request.state.inspect is not None:
        frame = request.state.inspect
        error_file = frame.f_code.co_filename
        error_func = frame.f_code.co_name
        error_line = frame.f_lineno
    else:
        error_func = error_file = error_line = "UNKNOWN"

    return {
        "errorFunc": error_func,
        "location": f"{error_line} line in {error_file}",
        "raised": str(error.__class__.__name__),
        "msg": str(error.ex),
    }


def api_logger(
    request: Request,
    response: Optional[Response] = None,
    error: Optional[InternalServerError | HTTPException | APIException] = None,
    **kwargs,
) -> None:
    processed_time = (
        time() - request.state.start if hasattr(request.state, "start") else -1
    )
    status_code = error.status_code if error else response.status_code
    user = request.state.user
    utc_now = datetime.utcnow()
    json_data = {
        "url": request.url.hostname + request.url.path,
        "method": str(request.method),
        "statusCode": status_code,
        "errorDetail": error_log_generator(error=error, request=request)
        if error is not None
        else None,
        "client": {
            "client": request.state.ip,
            "user": user.id if user and user.id else None,
            "email": hide_email(email=user.email)
            if user and user.email
            else None,
        },
        "processedTime": str(round(processed_time * 1000, 5)) + "ms",
        "datetimeUTC": utc_now.strftime("%Y/%m/%d %H:%M:%S"),
        "datetimeKST": (utc_now + timedelta(hours=9)).strftime("%Y/%m/%d %H:%M:%S"),
    } | kwargs
    log = json.dumps(json_data)
    logger.error(log) if error and error.status_code >= 500 else logger.info(log)
