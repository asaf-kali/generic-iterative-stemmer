from queue import Queue
from threading import Thread
from typing import Any, Callable, Iterable, Mapping, Optional


class AsyncTaskManager:
    def __init__(self, workers_amount: int = 5, iter_timeout: Optional[float] = 3):
        super().__init__()
        self.iter_timeout = iter_timeout
        self._task_queue: Queue = Queue()
        self._result_queue: Queue = Queue()
        self._total_task_count = 0
        self.start_workers(workers_amount)

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_done:
            raise StopIteration()
        return self.get_result(timeout=self.iter_timeout)

    @property
    def is_done(self) -> bool:
        return self._task_queue.unfinished_tasks == 0 and self._result_queue.unfinished_tasks == 0

    @property
    def total_task_count(self) -> int:
        return self._total_task_count

    def start_workers(self, workers_amount: int):
        for i in range(workers_amount):
            thread = Thread(target=self._work, daemon=True)
            thread.start()

    def _work(self):
        while True:
            func, args, kwargs = self._task_queue.get()
            result = func(*args, **kwargs)
            self._result_queue.put(result)
            self._task_queue.task_done()

    def add_task(self, func: Callable, args: Iterable = None, kwargs: Mapping = None):
        args = tuple(args) if args else ()
        kwargs = kwargs or {}
        self._total_task_count += 1
        self._task_queue.put((func, args, kwargs))

    def get_result(self, timeout: Optional[float] = 3) -> Any:
        result = self._result_queue.get(block=True, timeout=timeout)
        self._result_queue.task_done()
        return result

    def join(self):
        self._task_queue.join()
