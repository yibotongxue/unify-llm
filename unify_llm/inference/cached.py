import time
from collections.abc import Callable
from typing import Any

from ..cache_manager import BaseCacheManager
from ..utils.logger import Logger
from ..utils.tools import dict_to_hash
from ..utils.type_utils import InferenceInput, InferenceOutput
from .api_llm.base import BaseApiLLMInference
from .base import BaseInference, InferenceInterface


class CachedInference(InferenceInterface):
    """
    带缓存功能的推理类

    为推理结果提供缓存功能，减少重复计算，提高效率

    参数
    ----
    inference : BaseInference
        基础推理实例，用于执行实际的推理操作
    cache_manager : BaseCacheManager
        缓存管理器，用于存储和读取推理结果
    """

    def __init__(
        self, inference: BaseInference, cache_manager: BaseCacheManager
    ) -> None:
        self.inference = inference
        self.cache_manager = cache_manager
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def _generate(
        self,
        inputs: list[InferenceInput],
        enable_tqdm: bool = False,
        tqdm_args: dict[str, Any] | None = None,
    ) -> list[InferenceOutput]:
        """
        执行推理并利用缓存

        参数
        ----
        inputs : list[InferenceInput]
            输入数据列表
        enable_tqdm : bool, 默认为False
            是否显示进度条
        tqdm_args : dict[str, Any] | None, 默认为None
            进度条的参数配置

        返回
        ----
        list[InferenceOutput]
            推理结果列表，部分结果可能来自缓存
        """
        if isinstance(self.inference, BaseApiLLMInference):
            return self.inference.generate_with_cache(
                inputs, enable_tqdm, tqdm_args, self.cache_manager
            )
        cached_input_indices = []
        cached_result: list[InferenceOutput] = []
        for i, input in enumerate(inputs):
            input_key = self._generate_key(input)
            cache = self.cache_manager.load_cache(input_key)
            if (
                cache is not None
                and isinstance(cache, dict)
                and "data" in cache
                and isinstance(cache["data"], dict)
            ):
                cached_input_indices.append(i)
                cached_result.append(InferenceOutput(**cache["data"]))
        self.logger.info(
            f"一共有{len(inputs)}条请求，从缓存读取了{len(cached_input_indices)}条"
        )
        non_cached_inputs = [
            input for i, input in enumerate(inputs) if i not in cached_input_indices
        ]
        non_cached_outputs = self.inference._generate(
            non_cached_inputs, enable_tqdm, tqdm_args
        )
        for input, output in zip(non_cached_inputs, non_cached_outputs):
            key = self._generate_key(input)
            to_save_output = {
                "data": output.model_dump(),
                "meta_data": {
                    "time": time.time(),
                },
            }
            self.cache_manager.save_cache(key, to_save_output)
        cached_idx = 0
        non_cached_idx = 0
        result: list[InferenceOutput] = []
        for i in range(len(inputs)):
            if i in cached_input_indices:
                result.append(cached_result[cached_idx])
                cached_idx += 1
            else:
                result.append(non_cached_outputs[non_cached_idx])
                non_cached_idx += 1
        return result

    def _update_inference_cfgs(
        self, new_inference_cfgs: dict[str, Any]
    ) -> Callable[[], None]:
        """
        更新推理配置并返回恢复函数

        参数
        ----
        new_inference_cfgs : dict[str, Any]
            新的推理配置参数

        返回
        ----
        Callable[[], None]
            用于恢复原始配置的函数
        """
        return self.inference._update_inference_cfgs(new_inference_cfgs)

    def _generate_key(self, inference_input: InferenceInput) -> str:
        """
        为输入生成缓存键

        参数
        ----
        inference_input : InferenceInput
            输入数据

        返回
        ----
        str
            缓存键字符串
        """
        key_message = {
            "system_prompt": inference_input.system_prompt,
            "conversation": inference_input.conversation,
            "cfgs_hash": self.inference.inference_essential_cfgs_hash,
            "prefilled": inference_input.prefilled,
            "repeat_idx": inference_input.repeat_idx,
        }
        return dict_to_hash(key_message)
