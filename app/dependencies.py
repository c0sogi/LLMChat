from fastapi import Header, Query
from fastapi.security import APIKeyHeader


def api_service_dependency(
    secret: str = Header(...), key: str = Query(...), timestamp: str = Query(...)
):
    ...  # do some validation or processing with the headers


user_dependency = APIKeyHeader(name="Authorization", auto_error=False)
