# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Initializer registering the horizontal techniques scenario suite."""

from typing import ClassVar

from pyrit.models import DatasetSourceInput, ScenarioDatasetRequest, ScenarioPreset
from pyrit.setup.initializers.scenario_suites.scenario_suite_initializer import ScenarioSuiteInitializer


class HorizontalTechniquesSuiteInitializer(ScenarioSuiteInitializer):
    """
    Register the horizontal techniques suite, which runs one preset per attack technique family.

    Jailbreak templates, multilingual translation, encodings, and character obfuscation each run
    against the same objectives, so their success rates are directly comparable and show which
    technique families expose a target. Each family runs a small representative subset rather
    than every available technique, families do not overlap, and the baseline runs once, with
    the jailbreak family. The presets are hidden from the Scanner's list of individual scanners;
    launch them together from the Scanner's scenario suites table.
    """

    SUITE_NAME: ClassVar[str] = "horizontal_techniques"
    SUITE_DESCRIPTION: ClassVar[str] = (
        "Runs one representative preset per attack technique family (jailbreak templates, multilingual "
        "translation, encodings, and character obfuscation) against the same objectives, so per-family "
        "success rates are directly comparable."
    )

    DATASET_NAMES: ClassVar[tuple[str, ...]] = ("airt_horizontal_objectives",)
    JAILBREAK_TEMPLATES: ClassVar[tuple[str, ...]] = (
        "aim.yaml",
        "dan_11.yaml",
        "hypothetical_response.yaml",
        "prefix_injection.yaml",
        "refusal_suppression.yaml",
    )
    LANGUAGES: ClassVar[tuple[str, ...]] = ("Spanish", "Mandarin Chinese", "Bengali", "Swahili")
    ENCODING_TECHNIQUES: ClassVar[tuple[str, ...]] = ("base64", "binary", "morse", "rot13")
    OBFUSCATION_TECHNIQUES: ClassVar[tuple[str, ...]] = ("char_swap", "diacritic", "leetspeak", "unicode_confusable")

    def _build_presets(self) -> list[ScenarioPreset]:
        """
        Build one preset per technique family, all over the shared objectives.

        Returns:
            list[ScenarioPreset]: The jailbreak, multilingual, encoding, and obfuscation presets.
        """
        return [
            ScenarioPreset(
                name=f"{self.SUITE_NAME}_jailbreaks",
                scenario_name="airt.jailbreak",
                description="Renders a fixed set of jailbreak templates inline around each objective.",
                show_in_scanner=False,
                techniques=["prompt_sending"],
                datasets=self._shared_datasets(),
                include_baseline=True,
                scenario_params={"jailbreak_names": list(self.JAILBREAK_TEMPLATES)},
            ),
            ScenarioPreset(
                name=f"{self.SUITE_NAME}_multilingual",
                scenario_name="airt.multilingual",
                description="Translates each complete objective into a fixed set of languages.",
                show_in_scanner=False,
                techniques=["prompt_sending"],
                datasets=self._shared_datasets(),
                include_baseline=False,
                scenario_params={"languages": list(self.LANGUAGES), "translation_strategies": ["translation"]},
            ),
            ScenarioPreset(
                name=f"{self.SUITE_NAME}_encoding",
                scenario_name="foundry.red_team_agent",
                description="Encodes each objective with a fixed set of encodings and ciphers.",
                show_in_scanner=False,
                techniques=list(self.ENCODING_TECHNIQUES),
                datasets=self._shared_datasets(),
                include_baseline=False,
            ),
            ScenarioPreset(
                name=f"{self.SUITE_NAME}_obfuscation",
                scenario_name="foundry.red_team_agent",
                description="Perturbs the characters of each objective with a fixed set of obfuscations.",
                show_in_scanner=False,
                techniques=list(self.OBFUSCATION_TECHNIQUES),
                datasets=self._shared_datasets(),
                include_baseline=False,
            ),
        ]

    def _shared_datasets(self) -> ScenarioDatasetRequest:
        """
        Build the dataset request every preset runs, selecting every objective without sampling.

        Returns:
            ScenarioDatasetRequest: A new request over the shared objective datasets.
        """
        return ScenarioDatasetRequest(
            sources=[DatasetSourceInput(name=name) for name in self.DATASET_NAMES],
            max_per_dataset="all",
            max_total="all",
        )
