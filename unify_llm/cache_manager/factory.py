from typing import Any

from .base import BaseCacheManager
from .registry import CacheManagerRegistry


def get_cache_manager(cache_cfgs: dict[str, Any]) -> BaseCacheManager:
    cache_type = cache_cfgs.pop("cache_type", None)
    if cache_type is None:
        raise ValueError("The cache type should be set")
    return CacheManagerRegistry.get_by_name(cache_type)(cache_cfgs=cache_cfgs)
