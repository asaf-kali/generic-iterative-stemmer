from queue import Empty
from random import random
from time import sleep

import pytest

from generic_iterative_stemmer.utils.async_task_manager import AsyncTaskManager


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

    results = set(manager)

    assert results == inputs
    assert manager.is_done


def test_manager_raises_if_no_results_after_timeout():
    manager = AsyncTaskManager(workers_amount=1)
    with pytest.raises(Empty):
        _ = manager.get_result(timeout=0.05)
