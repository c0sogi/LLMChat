import asyncio
import queue
from collections import deque

from app.utils.system import free_memory_of_first_item_from_container

DEL_COUNT: int = 0


class DummyObject:
    """Dummy object for testing."""

    foo: bool
    bar: bool

    def __init__(self) -> None:
        """Initialize."""
        self.foo = True
        self.bar = False

    def __del__(self) -> None:
        """Clean up resources."""
        global DEL_COUNT
        DEL_COUNT += 1


def test_deallocate_item_from_memory_by_reference():
    """Tests if __del__ is called when item is removed from container."""
    global DEL_COUNT

    # Test with a deque
    _deque = deque([DummyObject() for _ in range(10)])
    _list = [DummyObject() for _ in range(10)]
    _dict = {i: DummyObject() for i in range(10)}
    _queue = queue.Queue()
    _asyncio_queue = asyncio.Queue()
    for _ in range(10):
        _queue.put(DummyObject())
        _asyncio_queue.put_nowait(DummyObject())

    # Test begin
    for container in [_deque, _list, _dict, _queue, _asyncio_queue]:
        print(f"- Testing {container.__class__.__name__}")
        DEL_COUNT = 0
        for _ in range(10):
            free_memory_of_first_item_from_container(container)
        print(f"- Finished testing {container.__class__.__name__}")
        assert (
            DEL_COUNT >= 10
        ), f"At least {container} should have 10 items deleted"


def test_dereference():
    obj = DummyObject()

    def delete_obj(obj) -> None:
        del obj

    def delete_foo(obj) -> None:
        del obj.foo

    delete_obj(obj)
    assert obj is not None  # Check if obj is still in memory
    delete_foo(obj)
    assert not hasattr(obj, "foo")  # Check if obj.foo is deleted
