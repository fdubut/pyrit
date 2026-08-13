# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the Multilingual scenario."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.converter import RandomTranslationConverter, TranslationConverter
from pyrit.models import AttackSeedGroup, ComponentIdentifier, SeedObjective
from pyrit.prompt_target import PromptTarget
from pyrit.scenario.scenarios.airt.multilingual import (
    _DEFAULT_NUM_LANGUAGES,
    _LANGUAGES_METADATA_KEY,
    Multilingual,
    _build_multilingual_technique,
)
from pyrit.score import TrueFalseScorer


def _mock_identifier(name: str) -> ComponentIdentifier:
    """Build a component identifier for a mock scenario dependency."""
    return ComponentIdentifier(class_name=name, class_module="test")


@pytest.fixture
def mock_memory_seed_groups() -> list[AttackSeedGroup]:
    """Create an inline objective population."""
    return [AttackSeedGroup(seeds=[SeedObjective(value="test objective")])]


@pytest.fixture
def mock_objective_target() -> PromptTarget:
    """Create the target under test."""
    mock = MagicMock(spec=PromptTarget)
    mock.get_identifier.return_value = _mock_identifier("MockObjectiveTarget")
    mock.configuration.includes.return_value = True
    return mock


@pytest.fixture
def mock_adversarial_chat() -> PromptTarget:
    """Create the target used by translation converters."""
    mock = MagicMock(spec=PromptTarget)
    mock.get_identifier.return_value = _mock_identifier("MockAdversarialChat")
    mock.capabilities.includes.return_value = True
    return mock


@pytest.fixture
def mock_objective_scorer() -> TrueFalseScorer:
    """Create the objective scorer."""
    mock = MagicMock(spec=TrueFalseScorer)
    mock.get_identifier.return_value = _mock_identifier("MockObjectiveScorer")
    return mock


def _patch_seed_groups(mock_memory_seed_groups):
    return patch.object(
        Multilingual,
        "_resolve_seed_groups_by_dataset_async",
        new_callable=AsyncMock,
        return_value={"inline": mock_memory_seed_groups},
    )


def _request_converter(atomic_attack):
    """Return the single request converter configured on an atomic attack."""
    configurations = atomic_attack.attack_technique.attack.get_request_converters()
    assert len(configurations) == 1
    assert len(configurations[0].converters) == 1
    return configurations[0].converters[0]


@pytest.mark.usefixtures("patch_central_database")
class TestMultilingual:
    """Validate multilingual technique selection and converter construction."""

    def test_technique_tags_define_aggregates(self) -> None:
        technique_class = _build_multilingual_technique()

        expected = {
            "prompt_sending",
            "random_translation",
        }
        assert {technique.value for technique in technique_class.expand({technique_class.SINGLE_TURN})} == expected

    def test_declares_run_parameters(self) -> None:
        """num_languages / languages are declared as run parameters."""
        names = {parameter.name for parameter in Multilingual.additional_parameters()}
        assert names == {"num_languages", "languages"}
        assert names.issubset({parameter.name for parameter in Multilingual.supported_parameters()})

    async def test_default_draws_two_random_languages(
        self, mock_objective_target, mock_adversarial_chat, mock_objective_scorer, mock_memory_seed_groups
    ):
        selected = ["French", "Spanish"]

        with (
            _patch_seed_groups(mock_memory_seed_groups),
            patch("pyrit.scenario.scenarios.airt.multilingual.random.sample", return_value=selected) as sample,
        ):
            scenario = Multilingual(
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=mock_objective_scorer,
            )
            scenario.set_params_from_args(args={"objective_target": mock_objective_target})
            await scenario.initialize_async()
            assert scenario._resolved_languages == selected
            assert sample.call_args.args[1] == _DEFAULT_NUM_LANGUAGES

    async def test_num_languages_samples_that_many(
        self, mock_objective_target, mock_adversarial_chat, mock_objective_scorer, mock_memory_seed_groups
    ):
        selected = ["French", "German", "Spanish"]

        with (
            _patch_seed_groups(mock_memory_seed_groups),
            patch("pyrit.scenario.scenarios.airt.multilingual.random.sample", return_value=selected) as sample,
        ):
            scenario = Multilingual(
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=mock_objective_scorer,
            )
            scenario.set_params_from_args(args={"objective_target": mock_objective_target, "num_languages": 3})
            await scenario.initialize_async()
            assert scenario._resolved_languages == selected
            assert sample.call_args.args[1] == 3

    async def test_explicit_languages_build_attacks_and_configure_random_translation(
        self, mock_objective_target, mock_adversarial_chat, mock_objective_scorer, mock_memory_seed_groups
    ):
        with _patch_seed_groups(mock_memory_seed_groups):
            scenario = Multilingual(
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=mock_objective_scorer,
            )
            scenario.set_params_from_args(
                args={"objective_target": mock_objective_target, "languages": ["Canadian French", "Spanish"]}
            )
            await scenario.initialize_async()

        assert [attack.atomic_attack_name for attack in scenario._atomic_attacks] == [
            "baseline",
            "translation_canadian_french",
            "translation_spanish",
            "random_translation",
        ]
        converters = [_request_converter(attack) for attack in scenario._atomic_attacks[1:4]]
        assert all(isinstance(converter, TranslationConverter) for converter in converters[0:2])
        assert isinstance(converters[2], RandomTranslationConverter)
        assert converters[2].languages == ["Canadian French", "Spanish"]

    async def test_mutually_exclusive_selectors_raise(
        self, mock_objective_target, mock_adversarial_chat, mock_objective_scorer, mock_memory_seed_groups
    ):
        with _patch_seed_groups(mock_memory_seed_groups):
            scenario = Multilingual(
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=mock_objective_scorer,
            )
            scenario.set_params_from_args(
                args={
                    "objective_target": mock_objective_target,
                    "num_languages": 2,
                    "languages": ["French"],
                }
            )
            with pytest.raises(ValueError, match="only one of"):
                await scenario.initialize_async()

    async def test_metadata_records_resolved_languages(
        self, mock_objective_target, mock_adversarial_chat, mock_objective_scorer, mock_memory_seed_groups
    ):
        with _patch_seed_groups(mock_memory_seed_groups):
            scenario = Multilingual(
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=mock_objective_scorer,
            )
            scenario.set_params_from_args(
                args={
                    "objective_target": mock_objective_target,
                    "languages": ["French", "Spanish"],
                }
            )
            await scenario.initialize_async()

        metadata = scenario._build_initial_scenario_metadata()
        assert metadata[_LANGUAGES_METADATA_KEY] == ["French", "Spanish"]

    def test_resolve_languages_replays_persisted_set_on_resume(self, mock_adversarial_chat, mock_objective_scorer):
        scenario = Multilingual(
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
            scenario_result_id="existing-result",
        )
        stored = MagicMock()
        stored.metadata = {_LANGUAGES_METADATA_KEY: ["French", "Spanish"]}

        with patch.object(scenario._memory, "get_scenario_results", return_value=[stored]):
            assert scenario._resolve_languages() == ["French", "Spanish"]

    async def test_baseline_is_prepended_by_default_with_same_seed_population(
        self, mock_objective_target, mock_adversarial_chat, mock_objective_scorer, mock_memory_seed_groups
    ):
        with _patch_seed_groups(mock_memory_seed_groups):
            scenario = Multilingual(
                adversarial_chat=mock_adversarial_chat,
                objective_scorer=mock_objective_scorer,
            )
            scenario.set_params_from_args(args={"objective_target": mock_objective_target, "languages": ["French"]})
            await scenario.initialize_async()

        assert scenario._atomic_attacks[0].atomic_attack_name == "baseline"
        assert scenario._atomic_attacks[0].seed_groups == scenario._atomic_attacks[1].seed_groups
        assert scenario._atomic_attacks[0].seed_groups[0] is scenario._atomic_attacks[1].seed_groups[0]
