from abc import ABC, abstractmethod
from typing import Any


class BaseCacheManager(ABC):
    def __init__(self, cache_cfgs: dict[str, Any]) -> None:
        self.cache_cfgs = cache_cfgs
        self.force_update = cache_cfgs.pop("force_update", False)

    def load_cache(self, key: str) -> dict[str, Any] | None:
        if self.force_update:
            return None
        return self._load_cache(key)

    @abstractmethod
    def _load_cache(self, key: str) -> dict[str, Any] | None:
        pass

    @abstractmethod
    def save_cache(self, key: str, value: dict[str, Any]) -> None:
        pass
