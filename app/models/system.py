from gc import collect
from logging import INFO, getLogger
from typing import TYPE_CHECKING, Any, Optional, Union
from collections import deque

if TYPE_CHECKING:
    from asyncio import Queue as AsyncQueue
    from logging import Logger
    from queue import Queue

ContainerLike = Union["deque", "Queue", "AsyncQueue", list, dict]


def get_vram_usages() -> Optional[list[int]]:
    """Returns a list of memory usage in MB for each GPU.
    Returns None if nvidia-smi is not available."""
    try:
        from subprocess import PIPE, run

        result = run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            stdout=PIPE,
        )
        return [
            int(mem)
            for mem in result.stdout.decode("utf-8").strip().split("\n")
        ]
    except Exception:
        return


def get_ram_usage() -> Optional[float]:
    """Returns the memory usage in MB.
    Returns None if psutil is not available."""
    try:
        from psutil import virtual_memory

        return virtual_memory().used / (1024**2)
    except Exception:
        return


def get_total_memory_usage() -> Optional[float]:
    """Returns the memory usage of RAM + VRAM in MB.
    Returns None if None of psutil and nvidia-smi are available."""
    vram_usages = get_vram_usages()
    ram_usage = get_ram_usage()
    if vram_usages is None and ram_usage is None:
        return
    elif vram_usages is None:
        return ram_usage
    elif ram_usage is None:
        return sum(vram_usages)
    else:
        return sum(vram_usages) + ram_usage


def deallocate_memory(item: Any) -> None:
    """Deallocate memory of the oldest object from container."""
    getattr(item, "__del__", lambda: None)()
    del item
    try:
        # Try to import empty_cache, which is only available in PyTorch
        from torch.cuda import empty_cache
    except ImportError:
        # If it fails, define an empty function
        empty_cache = lambda: None

    collect()  # Force garbage collection
    empty_cache()  # Empty VRAM cache


def free_memory_of_first_item_from_container(
    _container: ContainerLike,
    /,
    min_free_memory_mb: Optional[float] = None,
    logger: Optional["Logger"] = None,
) -> None:
    """Frees memory from a deque, list, or dict object by removing the first item.
    This function is useful when you want to deallocate memory.
    Proactively deallocating memory from a object can prevent memory leaks."""

    if logger is None:
        # If logger is not specified, create a new logger
        logger = getLogger(__name__)
        logger.setLevel(INFO)

    # Before creating a new completion generator, check memory usage
    mem_usage_before: Optional[float] = get_total_memory_usage()  # In MB
    if mem_usage_before is not None:
        logger.info(
            f"Deallocating memory from deque...\n- Current memory usage: {mem_usage_before} MB"
        )

    # Deallocate memory from the container
    if isinstance(_container, deque):
        item = _container.popleft()
    elif isinstance(_container, dict):
        item = _container.popitem()
    elif isinstance(_container, list):
        item = _container.pop(0)
    elif hasattr(_container, "get_nowait"):
        item = _container.get_nowait()
    elif hasattr(_container, "__getitem__") and hasattr(
        _container, "__delitem__"
    ):
        item = getattr(_container, "__getitem__")(0)
        getattr(_container, "__delitem__")(0)
    else:
        raise TypeError("Unsupported container type.")

    getattr(item, "__del__", lambda: None)()  # Invoke __del__ method forcibly
    del item
    try:
        # Try to import empty_cache, which is only available in PyTorch
        from torch.cuda import empty_cache
    except ImportError:
        # If it fails, define an empty function
        empty_cache = lambda: None

    collect()  # Force garbage collection
    empty_cache()  # Empty VRAM cache

    # And check memory usage again to see if there is a memory leak
    if mem_usage_before is not None:
        mem_usage_after = get_total_memory_usage()
        if mem_usage_after is not None:
            logger.info(
                (
                    f"Deallocated memory from deque.\n"
                    f"- Current memory usage: {mem_usage_after} MB"
                )
            )
            if (
                min_free_memory_mb is not None
                and mem_usage_before - mem_usage_after < min_free_memory_mb
            ):
                logger.warning(
                    (
                        f"RAM + VRAM usage did not decrease by at least {min_free_memory_mb} MB "
                        "after removing the oldest object.\n"
                        "This may indicate a memory leak.\n"
                        f"- Memory usage before: {mem_usage_before} MB\n"
                        f"- Memory usage after: {mem_usage_after} MB"
                    )
                )
                raise MemoryError("Memory leak occurred. Terminating...")
