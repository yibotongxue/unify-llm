from abc import ABC, abstractmethod
from typing import Any

from ..utils.type_utils import InferenceInput, InferenceOutput


class BasePromptBuilder(ABC):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def process_input_list(
        self, raw_inputs: list[InferenceInput]
    ) -> list[InferenceInput]:
        return [self.process_input(raw_input) for raw_input in raw_inputs]

    def parse_output_list(
        self, raw_outputs: list[InferenceOutput]
    ) -> list[InferenceOutput]:
        return [self.parse_output(raw_output) for raw_output in raw_outputs]

    @abstractmethod
    def process_input(self, raw_input: InferenceInput) -> InferenceInput:
        """
        Build a prompt from the raw prompt.

        Args:
            raw_prompt (InferenceInput): The raw prompt.

        Returns:
            InferenceInput: The built prompt.
        """

    @abstractmethod
    def parse_output(self, raw_output: InferenceOutput) -> InferenceOutput:
        """
        Parse the raw output to extract the answer.

        Args:
            raw_output (InferenceOutput): The raw output.

        Returns:
            InferenceOutput: The parsed output.
        """
