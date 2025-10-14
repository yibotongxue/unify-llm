import functools
import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def get_rank() -> int:
    """尝试从环境变量中获取当前进程的rank。"""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0


def rank_zero_only(fn: F) -> F:
    """装饰器：仅在rank=0时执行函数。"""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any | None:
        if get_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapper  # type: ignore
