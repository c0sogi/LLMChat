from asyncio import iscoroutinefunction
from functools import wraps
from time import time
from typing import Any, AsyncGenerator, Callable, Generator

from app.common.config import logging_config
from app.utils.logger import CustomLogger

logger = CustomLogger(
    name=__name__,
    logging_config=logging_config,
)


def log_time(fn: Any, elapsed: float) -> None:
    formatted_time = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}:{int((elapsed - int(elapsed)) * 1000000):06d}"
    logger.info(f"Function {fn.__name__} execution time: {formatted_time}")


def timeit(fn: Callable) -> Callable:
    """
    Decorator to measure execution time of a function, or generator.
    Supports both synchronous and asynchronous functions and generators.
    :param fn: function to measure execution time of
    :return: function wrapper

    Usage:
    >>> @timeit
    >>> def my_function():
    >>>     ...
    """

    @wraps(fn)
    async def coroutine_function_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time: float = time()
        fn_result = await fn(*args, **kwargs)
        elapsed: float = time() - start_time
        log_time(fn, elapsed)
        return fn_result

    @wraps(fn)
    def noncoroutine_function_wrapper(*args: Any, **kwargs: Any) -> Any:
        def sync_generator_wrapper() -> Generator:
            while True:
                try:
                    start_time: float = time()
                    item: Any = next(fn_result)
                    elapsed: float = time() - start_time
                    log_time(fn_result, elapsed)
                    yield item
                except StopIteration:
                    break

        async def async_generator_wrapper() -> AsyncGenerator:
            while True:
                try:
                    start_time: float = time()
                    item: Any = await anext(fn_result)
                    elapsed: float = time() - start_time
                    log_time(fn_result, elapsed)
                    yield item
                except StopAsyncIteration:
                    break

        start_time: float = time()
        fn_result: Any = fn(*args, **kwargs)
        elapsed: float = time() - start_time
        if isinstance(fn_result, Generator):
            return sync_generator_wrapper()
        elif isinstance(fn_result, AsyncGenerator):
            return async_generator_wrapper()
        log_time(fn, elapsed)
        return fn_result

    if iscoroutinefunction(fn):
        return coroutine_function_wrapper
    return noncoroutine_function_wrapper
