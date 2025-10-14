from __future__ import annotations

from copy import deepcopy
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict

from .logger import Logger, app_logger


class CustomBaseModel(BaseModel):  # type: ignore [misc]
    model_config = ConfigDict(extra="allow")

    def to_brief_dict(self) -> dict[str, Any]:
        raw_dict = deepcopy(self.model_dump())
        if "meta_data" in raw_dict:
            raw_dict.pop("meta_data")
        return raw_dict  # type: ignore [no-any-return]


class InferenceInput(CustomBaseModel):
    conversation: list[dict[str, Any]]
    prefilled: bool
    system_prompt: str
    ref_answer: str | None = None
    repeat_idx: int = 0
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_prompts(
        cls: type[InferenceInput], prompt: str, system_prompt: str = ""
    ) -> InferenceInput:
        return cls(
            conversation=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            prefilled=False,
            system_prompt=system_prompt,
            meta_data={},
        )

    @classmethod
    def from_output(
        cls,
        output: InferenceOutput,
        logger: Logger | None = None,
        use_parsed_output: bool = False,
    ) -> InferenceInput:
        conversation = InferenceInput(**output.input).conversation.copy()
        if conversation[0]["role"] == "system":
            conversation.pop(0)
        if conversation[-1]["role"] == "assistant":
            if logger is None:
                logger = app_logger
            logger.warning(
                f"The last turn of conversation is assistant.\nThe details is {conversation[-1]}. We will remove it"
            )
            conversation.pop()
        if use_parsed_output and output.parsed_output is not None:
            response = output.parsed_output
        else:
            response = output.response
        conversation.append({"role": "assistant", "content": response})
        return InferenceInput(
            conversation=conversation,
            prefilled=False,
            system_prompt="",
            meta_data={},
        )

    def get_last_user_message(self) -> str:
        if len(self.conversation) == 0:
            raise ValueError("The conversation is empty")
        if self.prefilled and len(self.conversation) == 1:
            raise ValueError("The conversation is prefilled, but only one turn")
        if self.prefilled:
            last_turn = self.conversation[-2]
        else:
            last_turn = self.conversation[-1]
        if last_turn["role"] != "user":
            raise ValueError("The last turn is not user")
        return last_turn["content"]  # type: ignore [no-any-return]

    def with_update_prompt(self, new_prompt: str) -> InferenceInput:
        new_conversation = deepcopy(self.conversation)
        if self.prefilled:
            new_conversation[-2] = {"role": "user", "content": new_prompt}
        else:
            new_conversation[-1] = {"role": "user", "content": new_prompt}
        raw = {
            **self.model_dump(),
            "conversation": new_conversation,
        }
        return InferenceInput(**raw)

    def get_raw_question(self) -> str:
        if "raw_question" in self.meta_data:
            return self.meta_data["raw_question"]  # type: ignore [no-any-return]
        if self.prefilled:
            return self.conversation[-2]["content"]  # type: ignore [no-any-return]
        return self.conversation[-1]["content"]  # type: ignore [no-any-return]

    def with_ref_answer(self, ref_answer: str) -> InferenceInput:
        raw = {
            **self.model_dump(),
            "ref_answer": ref_answer,
        }
        return InferenceInput(**raw)

    def with_system_prompt(self, system_prompt: str) -> InferenceInput:
        raw = {
            **self.model_dump(),
            "system_prompt": system_prompt,
        }
        return InferenceInput(**raw)

    def with_meta_data(self, meta_data: dict[str, Any]) -> InferenceInput:
        new_meta_data = {
            **self.meta_data,
            **meta_data,
        }
        raw = {
            **self.model_dump(),
            "meta_data": new_meta_data,
        }
        return InferenceInput(**raw)

    def with_prefill(self, prefix: str) -> InferenceInput:
        new_conversation = deepcopy(self.conversation)
        last_message = new_conversation[-1]
        if last_message["role"] == "assistant":
            last_message["content"] = prefix
        else:
            new_conversation.append({"role": "assistant", "content": prefix})
        raw = {
            **self.model_dump(),
            "conversation": new_conversation,
        }
        return InferenceInput(**raw)

    def with_repeat_idx(self, repeat_idx: int) -> InferenceInput:
        raw = {
            **self.model_dump(),
            "repeat_idx": repeat_idx,
        }
        return InferenceInput(**raw)


class InferenceOutput(CustomBaseModel):
    response: str
    parsed_output: Any | None = None
    input: dict[str, Any]
    engine: str
    meta_data: dict[str, Any]

    model_config = ConfigDict(extra="allow")

    def with_parsed_output(self, parsed_output: Any | None) -> InferenceOutput:
        raw = {
            **self.model_dump(),
            "parsed_output": parsed_output,
        }
        return InferenceOutput(**raw)


class InferSettings(TypedDict):
    repeat_cnt: int
    prompt_template: str | dict[str, Any] | None


def to_dict(obj: BaseModel | dict[str, Any]) -> dict[str, Any]:

    def _to_dict(
        obj: BaseModel | dict[str, Any] | list[Any] | Any
    ) -> dict[str, Any] | list[Any] | Any:
        if isinstance(obj, BaseModel):
            return _to_dict(obj.model_dump())
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_dict(e) for e in obj]
        return obj

    return _to_dict(obj)  # type: ignore [return-value]


def to_breif_dict(obj: CustomBaseModel | BaseModel | dict[str, Any]) -> dict[str, Any]:

    def _to_brief_dict(
        obj: CustomBaseModel | BaseModel | dict[str, Any] | list[Any] | Any
    ) -> dict[str, Any] | list[Any] | Any:
        if isinstance(obj, CustomBaseModel):
            return _to_brief_dict(obj.to_brief_dict())
        elif isinstance(obj, BaseModel):
            return _to_brief_dict(obj.model_dump())
        if isinstance(obj, dict):
            return {
                k: _to_brief_dict(v) for k, v in obj.items() if not k == "meta_data"
            }
        if isinstance(obj, list):
            return [_to_brief_dict(e) for e in obj]
        return obj

    return _to_brief_dict(obj)  # type: ignore [return-value]
