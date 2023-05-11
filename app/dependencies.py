from concurrent.futures import ProcessPoolExecutor
from fastapi import Header, Query
from fastapi.security import APIKeyHeader
from multiprocessing import Manager


def api_service_dependency(secret: str = Header(...), key: str = Query(...), timestamp: str = Query(...)):
    ...  # do some validation or processing with the headers


process_pool_executor = ProcessPoolExecutor()
process_manager = Manager()
user_dependency = APIKeyHeader(name="Authorization", auto_error=False)
