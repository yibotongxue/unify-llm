from typing import Any

from ..cache_manager import get_cache_manager
from ..utils.tools import dict_to_hash
from .base import BaseInference, InferenceInterface
from .cached import CachedInference


class InferenceFactory:
    """
    推理实例工厂类

    负责创建、缓存和管理推理实例，实现单例模式以避免重复创建相同配置的实例
    """

    _inference_pool: dict[str, BaseInference] = {}

    @classmethod
    def get_inference_instance(
        cls,
        model_cfgs: dict[str, Any],
        inference_cfgs: dict[str, Any],
        cache_cfgs: dict[str, Any] | None,
    ) -> InferenceInterface:
        """
        获取推理实例

        参数
        ----
        model_cfgs : dict[str, Any]
            模型配置参数
        inference_cfgs : dict[str, Any]
            推理配置参数
        cache_cfgs : dict[str, Any] | None
            缓存配置参数，如果为None则不启用缓存

        返回
        ----
        InferenceInterface
            推理实例，如果提供了缓存配置则返回带缓存功能的实例
        """
        instance = cls._get_inference_instance(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs
        )
        if cache_cfgs is None:
            return instance
        cache_manager = get_cache_manager(cache_cfgs=cache_cfgs)
        return CachedInference(inference=instance, cache_manager=cache_manager)

    @classmethod
    def _get_inference_instance(
        cls,
        model_cfgs: dict[str, Any],
        inference_cfgs: dict[str, Any],
    ) -> BaseInference:
        """
        获取或创建推理实例（内部方法）

        参数
        ----
        model_cfgs : dict[str, Any]
            模型配置参数
        inference_cfgs : dict[str, Any]
            推理配置参数

        返回
        ----
        BaseInference
            推理实例

        异常
        ----
        ValueError
            当指定的推理后端不受支持时抛出
        """
        cfgs_dict = {
            "model_cfgs": model_cfgs.copy(),
            "inference_cfgs": inference_cfgs.copy(),
        }
        cfgs_hash = dict_to_hash(cfgs_dict)

        backend = model_cfgs.get("inference_backend")

        # Shutdown previous instances if necessary
        if backend in ["hf", "vllm"]:
            for k, v in cls._inference_pool.items():
                if not k == cfgs_hash:
                    v.shutdown()

        if cfgs_hash in cls._inference_pool:
            return cls._inference_pool[cfgs_hash]

        backend = model_cfgs.get("inference_backend")

        instance: BaseInference | None = None

        if backend == "api":
            from .api_llm import get_api_llm_inference

            instance = get_api_llm_inference(
                model_cfgs=model_cfgs,
                inference_cfgs=inference_cfgs,
            )
        elif backend == "hf":
            from .hf import HuggingFaceInference

            instance = HuggingFaceInference(
                model_cfgs=model_cfgs,
                inference_cfgs=inference_cfgs,
            )
        elif backend == "vllm":
            from .vllm import VllmInference

            instance = VllmInference(
                model_cfgs=model_cfgs,
                inference_cfgs=inference_cfgs,
            )
        else:
            raise ValueError(f"Not supported inference backend: {backend}")
        cls._inference_pool[cfgs_hash] = instance

        return instance
