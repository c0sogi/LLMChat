import json
import logging
from typing import Union, Optional
from datetime import timedelta, datetime
from time import time
from fastapi.logger import logger
from starlette.requests import Request
from starlette.responses import Response

from app.errors.exceptions import APIException, SqlFailureEx


logger.setLevel(logging.INFO)


async def hide_email(email: str) -> str:
    separated_email = email.split("@")
    if len(separated_email) == 2:
        local_parts, domain = separated_email
        return "**" + local_parts[2:-1] + "*@" + domain[1]
    else:
        return "".join(separated_email)


async def error_log_generator(
    error: Union[SqlFailureEx, APIException], request: Request
):
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


async def api_logger(
    request: Request,
    response: Response = None,
    error: Optional[Union[SqlFailureEx, APIException]] = None,
    **kwargs,
):
    processed_time = time() - request.state.start
    status_code = error.status_code if error else response.status_code
    user = request.state.user
    utc_now = datetime.utcnow()

    log = json.dumps(
        {
            "url": request.url.hostname + request.url.path,
            "method": str(request.method),
            "statusCode": status_code,
            "errorDetail": await error_log_generator(error=error, request=request)
            if error is not None
            else None,
            "client": {
                "client": request.state.ip,
                "user": user.id if user and user.id else None,
                "email": (await hide_email(email=user.email))
                if user and user.email
                else None,
            },
            "processedTime": str(round(processed_time * 1000, 5)) + "ms",
            "datetimeUTC": utc_now.strftime("%Y/%m/%d %H:%M:%S"),
            "datetimeKST": (utc_now + timedelta(hours=9)).strftime("%Y/%m/%d %H:%M:%S"),
        }
        | kwargs
    )
    logger.error(log) if error and error.status_code >= 500 else logger.info(log)
