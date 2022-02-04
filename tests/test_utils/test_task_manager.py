from queue import Empty
from random import random
from time import sleep

import pytest

from generic_iterative_stemmer.utils import get_logger
from generic_iterative_stemmer.utils.async_task_manager import AsyncTaskManager

log = get_logger(__name__)


def dummy_task(x: int, duration: float = None):
    if not duration:
        duration = random() / 20
    sleep(duration)
    return x


def test_manager_is_results_iterable():
    manager = AsyncTaskManager(workers_amount=5)
    inputs = set(range(20))
    for i in inputs:
        manager.add_task(dummy_task, args=(i,))

    manager.join()
    results = set(manager)

    assert results == inputs
    assert manager.is_work_done
    assert manager.is_empty


def test_manager_is_a_context_manager():
    with AsyncTaskManager(workers_amount=5) as manager:
        inputs = set(range(20))
        for i in inputs:
            manager.add_task(dummy_task, args=(i,))
        results = set(manager)

        assert results == inputs
        assert manager.is_work_done
        assert manager.is_empty
        assert not manager.is_closed
    assert manager.is_closed


def test_manager_raises_empty_exception_if_no_results_after_timeout():
    manager = AsyncTaskManager(workers_amount=1)
    with pytest.raises(Empty):
        _ = manager.get_result(timeout=0.05)
    manager.join()


def test_no_tasks_are_processed_after_manager_joined():
    manager = AsyncTaskManager(workers_amount=5)
    inputs = set(range(50))
    for i in inputs:
        manager.add_task(dummy_task, args=(i,))
    manager.join()
    assert manager.is_closed

    manager.add_task(dummy_task, args=(999,))
    assert manager.is_work_done
    assert not manager.is_empty
    results = set(manager)
    assert manager.is_empty
    assert results == inputs
