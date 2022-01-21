from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass(frozen=True)
class FunctionCall:
    args: tuple
    kwargs: dict
    result: object


class FunctionHook:
    def __init__(self):
        self.calls: List[FunctionCall] = []

    def append(self, call: FunctionCall):
        self.calls.append(call)

    @property
    def results(self) -> list:
        return [call.result for call in self.calls]


def hook_calls(func: Callable) -> Tuple[Callable, FunctionHook]:
    hook = FunctionHook()

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        hook.append(FunctionCall(args, kwargs, result))
        return result

    return wrapper, hook
