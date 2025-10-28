import json
from pathlib import Path
from typing import Any

import redis  # type: ignore [import-untyped]
from redis.exceptions import (  # type: ignore [import-untyped]
    AuthenticationError,
    ConnectionError,
)

from ..utils.logger import Logger
from ..utils.multi_process import rank_zero_only
from .base import BaseCacheManager
from .registry import CacheManagerRegistry


@CacheManagerRegistry.register("redis")
class RedisCacheManager(BaseCacheManager):
    def __init__(self, cache_cfgs: dict[str, Any]) -> None:
        super().__init__(cache_cfgs=cache_cfgs)

        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        redis_cfgs = self.cache_cfgs.get("redis", {})

        # 添加连接池和重试机制
        pool = redis.ConnectionPool(
            max_connections=10,
            retry_on_timeout=True,
            socket_keepalive=True,
            **redis_cfgs,
        )
        self.redis_client = redis.Redis(connection_pool=pool)

        # 验证连接
        try:
            self.redis_client.ping()
        except AuthenticationError:
            self.logger.error(
                "Redis authentication failed. Please check your credentials."
            )
            raise RuntimeError("Redis authentication failed")
        except ConnectionError:
            self.logger.error(
                "Cannot connect to Redis. Please check your connection settings."
            )
            raise RuntimeError("Cannot connect to Redis")

        self._load_from_json_dir()

    @rank_zero_only
    def _load_from_json_dir(self) -> None:
        json_dir = self.cache_cfgs.get("json_dir", None)
        if json_dir is not None:
            self.logger.info(f"Loading cache from JSON files in {json_dir}")
            for json_file in Path(json_dir).glob("*.json"):
                with json_file.open("r", encoding="utf-8") as f:
                    content = json.load(f)
                    self.redis_client.set(json_file.stem, json.dumps(content))

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except redis.exceptions.ResponseError as e:
            self.logger.error(f"Redis load error: {str(e)}")
            return None

    def save_cache(self, key: str, value: dict[str, Any]) -> None:
        try:
            self.redis_client.set(
                key,
                json.dumps(value),
            )
        except redis.exceptions.ResponseError as e:
            self.logger.error(f"Redis save error: {str(e)}")
