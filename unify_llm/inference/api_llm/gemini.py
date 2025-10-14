import time
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import ContentDict

from ...utils.logger import Logger
from ...utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseApiLLMInference


class GeminiApiLLMInference(BaseApiLLMInference):
    """
    Google Gemini API接口的推理实现

    支持Google Gemini API接口协议的模型推理

    参数
    ----
    model_cfgs : dict[str, Any]
        模型配置参数，必须包含'model_name_or_path'和'api_key'
    inference_cfgs : dict[str, Any]
        推理配置参数
    """

    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        super().__init__(model_cfgs=model_cfgs, inference_cfgs=inference_cfgs)
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.model_name = self.model_cfgs["model_name_or_path"]
        api_key = self.model_cfgs["api_key"]
        self.client = genai.Client(api_key=api_key)

    def _single_generate(self, inference_input: InferenceInput) -> InferenceOutput:
        for i in range(self.max_retry):
            contents: list[ContentDict] = []
            for turn in inference_input.conversation:
                contents.append(
                    {"role": turn["role"], "parts": [{"text": turn["content"]}]}
                )
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=inference_input.system_prompt,
                        **self.inference_cfgs,
                    ),
                    contents=contents,
                )
            except Exception as err:
                self.logger.error(
                    msg=f"第{i+1}次呼叫{self.model_name} API失败，错误信息为{err}"
                )
                if i < self.max_retry - 1:
                    time.sleep(self.sleep_seconds)
                continue
            return InferenceOutput(
                response=response.text,
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
