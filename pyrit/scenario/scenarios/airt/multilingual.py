# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import random
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar

from pyrit.common import apply_defaults
from pyrit.common.path import DATASETS_PATH
from pyrit.converter import Converter, RandomTranslationConverter, TranslationConverter
from pyrit.executor.attack import AttackConverterConfig, AttackScoringConfig, PromptSendingAttack
from pyrit.models import Parameter, SeedDataset
from pyrit.prompt_normalizer import ConverterConfiguration
from pyrit.registry.components.attack_technique_registry import AttackTechniqueRegistry
from pyrit.scenario.core import (
    AtomicAttack,
    AttackTechnique,
    AttackTechniqueFactory,
    BaselineAttackPolicy,
    DatasetAttackConfiguration,
    Scenario,
    ScenarioTechnique,
    get_default_adversarial_target,
)
from pyrit.scenario.core.matrix_atomic_attack_builder import build_baseline_atomic_attack

if TYPE_CHECKING:
    from pyrit.models import AttackSeedGroup
    from pyrit.prompt_target import PromptTarget
    from pyrit.scenario.core import ScenarioTechnique
    from pyrit.scenario.core.scenario_context import ScenarioContext
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)

# Metadata key under which the resolved languages are persisted, so a resumed run
# replays the exact same set even when a random sample was drawn.
_LANGUAGES_METADATA_KEY = "languages"

# How many languages a bare run draws at random. Kept small so the default run stays fast
# — languages multiply against objectives and techniques. Override per run with
# ``num_languages`` (random count) or ``languages`` (an explicit set).
_DEFAULT_NUM_LANGUAGES = 2

#  Scenario-local default techniques.
#   - ``prompt_sending`` sends the objective in each selected language.
#   - ``random_translation`` sends the objective with word-level random translations.
_PROMPT_SENDING = "prompt_sending"
_RANDOM_TRANSLATION = "random_translation"


@cache
def _build_multilingual_technique() -> type[ScenarioTechnique]:
    """
    Build the Multilingual technique class from scenario-local factories.

    Returns:
        type[ScenarioTechnique]: The dynamically generated technique enum class.
    """
    factories = [
        AttackTechniqueFactory(
            name=_PROMPT_SENDING,
            attack_class=PromptSendingAttack,
            technique_tags=["single_turn"],
        ),
        AttackTechniqueFactory(
            name=_RANDOM_TRANSLATION,
            attack_class=PromptSendingAttack,
            technique_tags=["single_turn"],
        ),
    ]
    return AttackTechniqueRegistry.build_technique_class_from_factories(  # type: ignore[ty:invalid-return-type]
        class_name="MultilingualTechnique",
        factories=factories,
        default_tags={"single_turn"},
    )


class Multilingual(Scenario):
    """
    Multilingual scenario implementation for PyRIT.
    
    Tests how vulnerable a model is to non-English language use.
    """

    VERSION: int = 1
    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Enabled

    @classmethod
    def required_datasets(cls) -> list[str]:
        """Return a list of dataset names required by this scenario."""
        return ["harmbench"]

    @classmethod
    def additional_parameters(cls) -> list[Parameter]:
        """
        Declare the run-configurable parameters this scenario accepts (CLI / config file).

        Returns:
            list[Parameter]: The language selectors (``num_languages``, ``languages``).
        """
        return [
            Parameter(
                name="num_languages",
                description="Draw this many random languages. Mutually exclusive with languages.",
                param_type=int,
                default=None,
            ),
            Parameter(
                name="languages",
                description=(
                    "Explicit languages to use (e.g. French, German, Spanish). "
                    "When omitted, a random sample is drawn. Mutually exclusive with num_languages."
                ),
                param_type=list[str],
                default=None,
            ),
        ]

    @apply_defaults
    def __init__(
        self,
        *,
        adversarial_chat: PromptTarget | None = None,
        objective_scorer: TrueFalseScorer | None = None,
        scenario_result_id: str | None = None,
    ) -> None:
        """
        Initialize the multilingual scenario.

        Args:
            adversarial_chat (PromptTarget | None): Target used by the translation converters.
            objective_scorer (TrueFalseScorer | None): Scorer used to evaluate target responses.
            scenario_result_id (str | None): Optional ID of an existing scenario result to resume.
        """
        self._adversarial_chat = adversarial_chat
        self._objective_scorer: TrueFalseScorer = (
            objective_scorer if objective_scorer else self._get_default_objective_scorer()
        )
        self._default_languages = self._get_default_languages()
        self._resolved_languages: list[str] = []

        technique_class = _build_multilingual_technique()

        super().__init__(
            version=self.VERSION,
            technique_class=technique_class,
            default_dataset_config=DatasetAttackConfiguration(dataset_names=["harmbench"], max_dataset_size=4),
            objective_scorer=self._objective_scorer,
            scenario_result_id=scenario_result_id,
        )

    @classmethod
    def _get_default_languages(cls) -> list[str]:
        """
        Load the default languages from the public PyRIT lexicon.

        Returns:
            list[str]: The list of most-spoken languages.
        """
        dataset = SeedDataset.from_yaml_file(DATASETS_PATH / "lexicons" / "languages_most_spoken.yaml")
        return [str(seed.value) for seed in dataset.seeds]

    def _resolve_languages(self) -> list[str]:
        """
        Resolve the languages for this run, replaying the persisted set on resume.

        On a fresh run this reads the run parameters: an explicit ``languages`` set or a random
        ``num_languages`` sample (defaulting to a small random draw when neither is given). On resume
        the originally chosen set is read back from the stored ``ScenarioResult`` metadata so a random
        sample isn't redrawn (which would diverge from the persisted attacks).

        Returns:
            list[str]: The explicit or randomly sampled languages for this run.

        Raises:
            ValueError: If both ``num_languages`` and ``languages`` are provided,
            or if ``num_languages`` is out of bounds.
        """
        if self._scenario_result_id is not None:
            stored = self._memory.get_scenario_results(scenario_result_ids=[self._scenario_result_id])
            if stored:
                persisted = (stored[0].metadata or {}).get(_LANGUAGES_METADATA_KEY)
                if persisted:
                    return list(persisted)

        num_languages = self.params.get("num_languages")
        languages = self.params.get("languages")

        if num_languages and languages:
            raise ValueError(
                "Please provide only one of `num_languages` (random selection)"
                " or `languages` (specific selection)."
            )

        if languages:
            return languages

        count = int(num_languages) if num_languages is not None else _DEFAULT_NUM_LANGUAGES
        if count < 1 or count > len(self._default_languages):
            raise ValueError(f"num_languages must be between 1 and {len(self._default_languages)}.")
        return random.sample(self._default_languages, count)

    def _build_initial_scenario_metadata(self) -> dict[str, Any]:
        """
        Persist the resolved languages alongside the base scenario metadata.

        Returns:
            dict[str, Any]: The base metadata plus the resolved language set.
        """
        metadata = super()._build_initial_scenario_metadata()
        metadata[_LANGUAGES_METADATA_KEY] = list(self._resolved_languages)
        return metadata

    async def _build_atomic_attacks_async(self, *, context: ScenarioContext) -> list[AtomicAttack]:
        """
        Build the selected translation attacks over the resolved objective population.

        Args:
            context (ScenarioContext): The resolved runtime inputs for this run.

        Returns:
            list[AtomicAttack]: The atomic attacks to execute.

        Raises:
            ValueError: If the scenario is not properly initialized.
        """
        if self._objective_target is None:
            raise ValueError(
                "Scenario not properly initialized. Call await scenario.initialize_async() before running."
            )

        self._resolved_languages = self._resolve_languages()
        adversarial_chat = self._adversarial_chat or get_default_adversarial_target()
        techniques = {technique.value for technique in context.scenario_techniques}
        seed_groups = list(context.seed_groups)

        atomic_attacks: list[AtomicAttack] = []
        if context.include_baseline:
            atomic_attacks.append(
                build_baseline_atomic_attack(
                    objective_target=context.objective_target,
                    objective_scorer=self._objective_scorer,
                    seed_groups=seed_groups,
                    memory_labels=context.memory_labels,
                )
            )

        if _PROMPT_SENDING in techniques:
            atomic_attacks.extend(
                self._build_atomic_attack(
                    context=context,
                    seed_groups=seed_groups,
                    converter=TranslationConverter(converter_target=adversarial_chat, language=language),
                    name=f"translation_{language.lower().replace(' ', '_')}",
                    display_group=language,
                )
                for language in self._resolved_languages
            )

        if _RANDOM_TRANSLATION in techniques:
            atomic_attacks.append(
                self._build_atomic_attack(
                    context=context,
                    seed_groups=seed_groups,
                    converter=RandomTranslationConverter(
                        converter_target=adversarial_chat,
                        languages=self._resolved_languages,
                    ),
                    name="random_translation",
                    display_group="Random Translation",
                )
            )

        return atomic_attacks

    def _build_atomic_attack(
        self,
        *,
        context: ScenarioContext,
        seed_groups: list[AttackSeedGroup],
        converter: Converter,
        name: str,
        display_group: str,
    ) -> AtomicAttack:
        """
        Build a prompt-sending atomic attack with one request converter.

        Returns:
            AtomicAttack: The configured attack and its resolved seed groups.
        """
        converter_config = AttackConverterConfig(
            request_converters=ConverterConfiguration.from_converters(converters=[converter])
        )
        attack = PromptSendingAttack(
            objective_target=context.objective_target,
            attack_scoring_config=AttackScoringConfig(objective_scorer=self._objective_scorer),
            attack_converter_config=converter_config,
        )
        return AtomicAttack(
            atomic_attack_name=name,
            display_group=display_group,
            attack_technique=AttackTechnique(attack=attack),
            seed_groups=seed_groups,
            objective_scorer=self._objective_scorer,
            memory_labels=context.memory_labels,
        )
