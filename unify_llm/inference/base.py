from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, ContextManager

from ..prompts import BasePromptBuilder, PromptBuilderRegistry
from ..utils.config import deepcopy_config
from ..utils.shutdownable import Shutdownable
from ..utils.tools import dict_to_hash
from ..utils.type_utils import InferenceInput, InferenceOutput


class InferenceInterface(ABC):
    """
    推理接口的抽象基类

    定义了模型推理的基本接口，提供了生成结果、更新配置等通用功能
    """

    def generate(
        self,
        inputs: list[InferenceInput],
        *,
        repeat_cnt: int = 1,
        prompt_template: str | dict[str, Any] | None = None,
        enable_tqdm: bool = False,
        tqdm_args: dict[str, Any] | None = None,
    ) -> list[list[InferenceOutput]]:
        """
        生成推理结果

        参数
        ----
        inputs : list[InferenceInput]
            输入数据列表
        repeat_cnt : int, 默认为1
            每个输入重复推理的次数
        prompt_template : str | None, 默认为None
            提示模板名称，用于构建或修改提示
        enable_tqdm : bool, 默认为False
            是否显示进度条
        tqdm_args : dict[str, Any] | None, 默认为None
            进度条的参数配置

        返回
        ----
        list[list[InferenceOutput]]
            每个输入对应的推理结果列表，内层列表包含重复次数的结果
        """
        prompt_builder: BasePromptBuilder | None = None
        if isinstance(prompt_template, str):
            prompt_builder = PromptBuilderRegistry.get_by_name(prompt_template)()
        if isinstance(prompt_template, dict):
            prompt_builder_name = prompt_template.pop("name")
            prompt_builder = PromptBuilderRegistry.get_by_name(prompt_builder_name)(
                config=prompt_template
            )

        if prompt_builder is not None:
            inputs = [
                InferenceInput(**deepcopy_config(input.model_dump()))
                for input in inputs
            ]
            inputs = prompt_builder.process_input_list(inputs)
        repeat_inputs: list[InferenceInput] = []
        for input in inputs:
            for repeat_idx in range(repeat_cnt):
                repeat_inputs.append(input.with_repeat_idx(repeat_idx))
        outputs = self._generate(
            repeat_inputs, enable_tqdm=enable_tqdm, tqdm_args=tqdm_args
        )
        if prompt_builder is not None:
            outputs = prompt_builder.parse_output_list(outputs)
        grouped_outputs = [
            outputs[i : i + repeat_cnt] for i in range(0, len(outputs), repeat_cnt)
        ]
        return grouped_outputs

    @abstractmethod
    def _generate(
        self,
        inputs: list[InferenceInput],
        enable_tqdm: bool = False,
        tqdm_args: dict[str, Any] | None = None,
    ) -> list[InferenceOutput]:
        """
        内部生成方法，由子类实现

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
            推理结果列表

        注意
        ----
        此方法必须由子类实现
        """

    def update_inference_cfgs(
        self, new_inference_cfgs: dict[str, Any]
    ) -> ContextManager[InferenceInterface]:
        """
        临时更新推理配置

        参数
        ----
        new_inference_cfgs : dict[str, Any]
            新的推理配置参数

        返回
        ----
        ContextManager[InferenceInterface]
            上下文管理器，用于临时应用配置并在退出时恢复原配置
        """
        return self._TempConfigUpdater(self, new_inference_cfgs)

    @abstractmethod
    def _update_inference_cfgs(
        self, new_inference_cfgs: dict[str, Any]
    ) -> Callable[[], None]:
        """
        内部更新配置方法，由子类实现

        参数
        ----
        new_inference_cfgs : dict[str, Any]
            新的推理配置参数

        返回
        ----
        Callable[[], None]
            用于恢复原始配置的函数

        注意
        ----
        此方法必须由子类实现
        """

    class _TempConfigUpdater(AbstractContextManager["InferenceInterface"]):
        """
        内部上下文管理器类，处理配置的临时更新和恢复

        用于在上下文范围内临时修改推理配置，退出上下文后自动恢复原配置
        """

        def __init__(
            self, owner: InferenceInterface, new_inference_cfgs: dict[str, Any]
        ):
            self.owner = owner  # 主类实例
            self.new_inference_cfgs = new_inference_cfgs  # 要更新的配置
            self.restore_func: Callable[[], None] | None = None

        def __enter__(self) -> InferenceInterface:
            """进入上下文时：备份原始配置并应用新配置"""
            self.restore_func = self.owner._update_inference_cfgs(
                self.new_inference_cfgs
            )

            return self.owner  # 返回主类实例以便链式调用

        def __exit__(self, exc_type, exc_value, traceback):  # type: ignore [no-untyped-def]
            """退出上下文时：恢复原始配置"""
            # 无论是否发生异常都恢复配置
            if self.restore_func is not None:
                self.restore_func()
            # 不处理异常，返回 None 让异常正常传播


class BaseInference(InferenceInterface, Shutdownable):
    """
    基础推理实现类

    实现了推理接口的基本功能，提供配置管理、哈希计算等通用功能

    参数
    ----
    model_cfgs : dict[str, Any]
        模型配置参数
    inference_cfgs : dict[str, Any]
        推理配置参数
    """

    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        cfgs_dict = {
            "model_cfgs": model_cfgs,
            "inference_cfgs": inference_cfgs,
        }
        self._cfgs_hash = dict_to_hash(cfgs_dict)
        self.model_cfgs = model_cfgs
        self.inference_cfgs = inference_cfgs
        self._inference_essential_cfgs_hash = dict_to_hash(
            self._get_inference_essential_cfgs()
        )

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
        self.original_inference_cfgs = self.inference_cfgs.copy()
        self.original_cfgs_hash = self._cfgs_hash
        self.inference_cfgs.update(new_inference_cfgs)
        self._cfgs_hash = dict_to_hash(
            {
                "model_cfgs": self.model_cfgs,
                "inference_cfgs": self.inference_cfgs,
            }
        )

        def _restore_inference_cfgs() -> None:
            """Restore the original inference configurations."""
            self.inference_cfgs = self.original_inference_cfgs
            self._cfgs_hash = self.original_cfgs_hash

        return _restore_inference_cfgs

    @property
    def cfgs_hash(self) -> str:
        return self._cfgs_hash

    @property
    def inference_essential_cfgs_hash(self) -> str:
        return self._inference_essential_cfgs_hash

    def _get_inference_essential_cfgs(self) -> dict[str, Any]:
        """
        获取推理的基本配置

        返回
        ----
        dict[str, Any]
            包含模型和推理配置的字典
        """
        return {
            "model_cfgs": self.model_cfgs,
            "inference_cfgs": self.inference_cfgs,
        }
