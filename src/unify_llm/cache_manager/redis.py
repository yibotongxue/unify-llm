import json
import time
from typing import Any

import redis  # type: ignore [import-untyped]
from redis.exceptions import (  # type: ignore [import-untyped]
    AuthenticationError,
    ConnectionError,
)

from ..utils.logger import Logger
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

        self.max_retries: int = redis_cfgs.get("max_retries", 3)
        self.sleep_interval: float = redis_cfgs.get("sleep_interval", 0.1)

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        for attempt in range(self.max_retries):
            try:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                self.logger.error(
                    f"Redis load error: {str(e)} in attempt {attempt + 1}"
                )
                time.sleep(self.sleep_interval)
        self.logger.error(
            f"Failed to load cache for key: {key} after {self.max_retries} attempts"
        )
        return None

    def save_cache(self, key: str, value: dict[str, Any]) -> None:
        for attempt in range(self.max_retries):
            try:
                self.redis_client.set(
                    key,
                    json.dumps(value),
                )
                return
            except Exception as e:
                self.logger.error(
                    f"Redis save error: {str(e)} in attempt {attempt + 1}"
                )
                time.sleep(self.sleep_interval)
        self.logger.error(
            f"Failed to save cache for key: {key} after {self.max_retries} attempts"
        )
