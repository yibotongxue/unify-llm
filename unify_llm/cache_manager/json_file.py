from pathlib import Path
from typing import Any

from ..utils.json_utils import load_json, save_json
from ..utils.logger import Logger
from .base import BaseCacheManager
from .registry import CacheManagerRegistry


@CacheManagerRegistry.register("json_file")
class JSONFileCacheManager(BaseCacheManager):
    def __init__(self, cache_cfgs: dict[str, Any]) -> None:
        super().__init__(cache_cfgs)
        self.cache_dir = Path(cache_cfgs.get("cache_dir", "./cache"))
        self.flush_threshold = cache_cfgs.get("flush_threshold", 10)
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._dirty_keys: set[str] = set()

        self._load_all_from_disk()

    def _safe_key(self, key: str) -> str:
        return key.replace("/", "_")  # 简单处理非法文件名字符

    def _get_file_path(self, key: str) -> Path:
        return self.cache_dir / f"{self._safe_key(key)}.json"

    def _load_all_from_disk(self) -> None:
        for file in self.cache_dir.glob("*.json"):
            key = file.stem
            try:
                self._memory_cache[key] = load_json(str(file))
            except Exception as err:
                self.logger.warning(
                    f"读取文件{str(file)}的时候出现异常，异常信息为{err}"
                )
                continue  # 忽略损坏的文件

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        return self._memory_cache.get(key)

    def save_cache(self, key: str, value: dict[str, Any]) -> None:
        self._memory_cache[key] = value
        self._dirty_keys.add(key)

        if len(self._dirty_keys) >= self.flush_threshold:
            self._flush_dirty_to_disk()

    def _flush_dirty_to_disk(self) -> None:
        for key in list(self._dirty_keys):
            path = self._get_file_path(key)
            save_json(self._memory_cache[key], str(path))
        self._dirty_keys.clear()

    def __del__(self) -> None:
        self._flush_dirty_to_disk()
