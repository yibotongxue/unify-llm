import time
from typing import Any

import anthropic
from anthropic.types.message_param import MessageParam

from ...utils.logger import Logger
from ...utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseApiLLMInference


class AnthropicApiLLMInference(BaseApiLLMInference):
    """
    Anthropic Claude API接口的推理实现

    支持Anthropic API接口协议的模型推理，包括Claude系列及兼容接口

    参数
    ----
    model_cfgs : dict[str, Any]
        模型配置参数，必须包含'model_name_or_path'和'api_key'
    inference_cfgs : dict[str, Any]
        推理配置参数，可包含max_tokens等
    """

    _BASE_URL_MAP: dict[str, str] = {
        "kimi-k2-0711-preview": "https://api.moonshot.cn/anthropic",
    }

    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        super().__init__(model_cfgs=model_cfgs, inference_cfgs=inference_cfgs)
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.model_name = self.model_cfgs["model_name_or_path"]
        api_key = self.model_cfgs["api_key"]
        base_url = self._BASE_URL_MAP.get(self.model_name, None)
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self.max_tokens = self.inference_cfgs.pop("max_tokens", 1024)

    def _single_generate(self, inference_input: InferenceInput) -> InferenceOutput:
        messages: list[MessageParam] = []
        messages.append(
            {
                "role": "system",
                "content": inference_input.system_prompt,
            }
        )
        for turn in inference_input.conversation:
            messages.append(
                {
                    "role": turn["role"],
                    "content": turn["content"],
                }
            )
        for i in range(self.max_retry):
            try:
                response = self.client.messages.create(
                    max_tokens=self.max_tokens,
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    **self.inference_cfgs,
                )
            except Exception as err:
                self.logger.error(
                    msg=f"第{i+1}次呼叫{self.model_name} API失败，错误信息为{err}"
                )
                if i < self.max_retry - 1:
                    time.sleep(self.sleep_seconds)
                continue
            content = response.content[0].text
            return InferenceOutput(
                response=content,
                input=inference_input.model_dump(),
                engine="api",
                meta_data={
                    "raw_output": response.model_dump(),
                    "model_cfgs": self.safe_model_cfgs,
                    "inference_cfgs": self.inference_cfgs,
                },
            )
        self.logger.error(
            msg=f"所有对{self.model_name} API的呼叫均以失败，返回默认信息"
        )
        return InferenceOutput(
            response="",
            input=inference_input.model_dump(),
            engine="api",
            meta_data={
                "error": "All API calls failed",
                "model_cfgs": self.safe_model_cfgs,
                "inference_cfgs": self.inference_cfgs,
            },
        )

    @classmethod
    def register_model(cls, model_name: str, base_url: str) -> None:
        """
        注册新的模型与基础URL映射

        参数
        ----
        model_name : str
            模型名称
        base_url : str
            API基础URL
        """
        cls._BASE_URL_MAP[model_name] = base_url
