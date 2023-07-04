from gc import collect
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from collections import deque
    from logging import Logger


def get_vram_usages() -> Optional[list[int]]:
    """Returns a list of memory usage in MB for each GPU.
    Returns None if nvidia-smi is not available."""
    try:
        from subprocess import PIPE, run

        result = run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            stdout=PIPE,
        )
        return [int(mem) for mem in result.stdout.decode("utf-8").strip().split("\n")]
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


def free_memory_from_deque(
    deque_object: "deque",
    min_free_memory_mb: float = 512,
    logger: Optional["Logger"] = None,
) -> None:
    try:
        from torch.cuda import empty_cache
    except Exception:
        empty_cache = None

    # Before creating a new completion generator, check memory usage
    mem_usage_before: Optional[float] = get_total_memory_usage()  # In MB
    if logger is not None and mem_usage_before is not None:
        logger.info(
            f"Deallocating memory from deque...\n- Current memory usage: {mem_usage_before} MB"
        )

    # Remove the first object from the deque
    # And check memory usage again to see if there is a memory leak
    (deque_object.popleft()).__del__()
    collect()
    if empty_cache is not None:
        empty_cache()

    if mem_usage_before is not None:
        mem_usage_after = get_total_memory_usage()
        if mem_usage_after is not None:
            if logger is not None:
                logger.info(
                    (
                        f"Deallocated memory from deque.\n"
                        f"- Current memory usage: {mem_usage_after} MB"
                    )
                )
            if mem_usage_before - mem_usage_after < min_free_memory_mb:
                if logger is not None:
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
