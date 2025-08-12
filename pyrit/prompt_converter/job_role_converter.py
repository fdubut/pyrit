# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptDataType, SeedPrompt
from pyrit.prompt_converter import ConverterResult, LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class JobRoleGenerator(LLMGenericTextConverter):
    """
    Tests the implicit bias of an LLM related to the demographics of specific job roles.

    Based on Project Moonshot: https://github.com/aiverify-foundation/moonshot-data
    """

    def __init__(
        self,
        *,
        converter_target: PromptChatTarget,
        prompt_template: Optional[SeedPrompt] = None,
        job: str,
    ):
        """
        Initializes the converter with a specific target, job role and template.

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt.
            prompt_template (SeedPrompt): The prompt template to use.
            job (str): The job role to append demographic groups to.
        """
        # Set to default strategy if not provided
        self._prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "job_role_converter.yaml"
            )
        )

        super().__init__(converter_target=converter_target, prompt_template=self._prompt_template, job=job)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Appends a demographic group to the job role.

        Parameters:
            prompt (str): The demographic group to append.
            input_type (PromptDataType): The type of input data.

        Returns:
            ConverterResult: The result of the conversion, including the job role and demographic group.

        Raises:
            ValueError: If the input type is not supported.
        """
        return await super().convert_async(prompt=prompt, input_type=input_type)
