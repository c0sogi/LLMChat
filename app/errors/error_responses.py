from dataclasses import dataclass
from fastapi import HTTPException


@dataclass(frozen=True)
class ErrorResponses:
    no_email_or_password: HTTPException = HTTPException(
        400, "Email and PW must be provided."
    )
    email_already_taken: HTTPException = HTTPException(400, "Email already exists.")
    not_supported_feature: HTTPException = HTTPException(400, "Not supported.")
    no_matched_user: HTTPException = HTTPException(400, "No matched user.")
    enforce_domain_wildcard: HTTPException = HTTPException(
        500, "Domain wildcard patterns must be like '*.example.com'."
    )
