from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T", bound=object)


class BaseRegistry(Generic[T]):
    """通用注册表基类"""

    # 基类不定义_registry，由子类各自拥有

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """子类初始化时创建独立的注册表"""
        super().__init_subclass__(**kwargs)
        # 每个子类创建自己的_registry（覆盖基类可能存在的属性）
        cls._registry: dict[str, type[T]] = {}  # type: ignore [misc, attr-defined]

    @classmethod
    def register(cls, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(subclass: type[T]) -> type[T]:
            # 注册到调用该方法的类自己的_registry中
            cls._registry[name] = subclass  # type: ignore [attr-defined]
            return subclass

        return decorator

    @classmethod
    def get_by_name(cls, name: str) -> type[T]:
        if name not in cls._registry:  # type: ignore [attr-defined]
            raise KeyError(f"No class registered under '{name}' in {cls.__name__}")
        return cls._registry[name]  # type: ignore [attr-defined, no-any-return]
