import os
import time
from typing import Any

from dashscope import MultiModalConversation

from ...utils.logger import Logger
from ...utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseApiLLMInference


class DashScopeLLMInference(BaseApiLLMInference):
    """
    DashScope API接口的推理实现
    """

    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        super().__init__(model_cfgs=model_cfgs, inference_cfgs=inference_cfgs)
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.model_name = self.model_cfgs["model_name_or_path"]
        self.api_key = self.model_cfgs["api_key"]

    def _prepare_input(self, inference_input: InferenceInput) -> InferenceInput:
        for turn in inference_input.conversation:
            if isinstance(turn["content"], str):
                turn["content"] = [{"text": turn["content"]}]
            elif isinstance(turn["content"], list):
                new_content = []
                for item in turn["content"]:
                    if isinstance(item, str):
                        new_content.append({"text": item})
                    elif isinstance(item, dict):
                        if "image" in item and not item["image"].startswith("http"):
                            image_file_path = item["image"]
                            # convert relative path to absolute path
                            if not os.path.isabs(image_file_path):
                                image_file_path = os.path.join(
                                    os.getcwd(), image_file_path
                                )
                            item["image"] = f"file://{image_file_path}"
                        new_content.append(item)
                turn["content"] = new_content
        return inference_input

    def _single_generate(self, inference_input: InferenceInput) -> InferenceOutput:
        inference_input = self._prepare_input(inference_input)
        messages: list[dict[str, Any]] = []
        messages.append(
            {
                "role": "system",
                "content": [{"text": inference_input.system_prompt}],
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
                response = MultiModalConversation.call(
                    model=self.model_name,
                    messages=messages,
                    api_key=self.api_key,
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
            content = response["output"]["choices"][0]["message"]["content"][0]["text"]
            return InferenceOutput(
                response=content,
                input=inference_input.model_dump(),
                engine="api",
                meta_data={
                    "raw_output": response,
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


def main() -> None:
    import argparse

    from ...utils.config import load_config, update_config_with_unparsed_args
    from ...utils.type_utils import InferenceInput
    from .factory import get_api_llm_inference

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file-path",
        type=str,
        required=True,
        help="The path to the config file",
    )
    args, unparsed_args = parser.parse_known_args()

    cfgs = load_config(args.config_file_path)
    update_config_with_unparsed_args(unparsed_args=unparsed_args, cfgs=cfgs)

    inference = get_api_llm_inference(cfgs["model_cfgs"], cfgs["inference_cfgs"])

    inference_input = [
        InferenceInput(
            conversation=[{"role": "user", "content": "中国的首都是哪里？"}],
            prefilled=False,
            system_prompt="你是一个人工智能助手",
            meta_data={},
        ),
        InferenceInput(
            conversation=[
                {
                    "role": "user",
                    "content": [
                        {"text": "请描述下面的图片内容"},
                        {"image": "../../Downloads/dog_and_girl.jpeg"},
                    ],
                }
            ],
            prefilled=False,
            system_prompt="你是一个人工智能助手",
            meta_data={},
        ),
    ]

    outputs = inference._generate(
        inference_input,
        enable_tqdm=True,
        tqdm_args={"desc": "Generating responses using api"},
    )

    for inference_output in outputs:
        print(inference_output.response)


if __name__ == "__main__":
    main()
