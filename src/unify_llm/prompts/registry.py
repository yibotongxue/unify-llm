from typing import Any

from ..utils.registry import BaseRegistry
from .base import BasePromptBuilder


class PromptBuilderRegistry(BaseRegistry[BasePromptBuilder]):
    @classmethod
    def verify_type(
        cls, name: str | dict[str, Any], type_bound: type[BasePromptBuilder]
    ) -> None:
        if isinstance(name, str):
            instance = cls.get_by_name(name)()
        elif isinstance(name, dict):
            name = name.copy()
            prompt_builder_name = name.pop("name")
            instance = cls.get_by_name(prompt_builder_name)(config=name)
        else:
            raise TypeError(f"提示词配置必须为字符串或字典")
        if not isinstance(instance, type_bound):
            raise ValueError(
                f"Prompt builder type '{name}' is not a valid {type_bound.__name__}."
            )
