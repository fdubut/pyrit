# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Code-registered scenario suites built by suite initializers."""

import itertools
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

import pytest

from pyrit.datasets.seed_datasets.local.local_dataset_loader import _LocalDatasetLoader
from pyrit.models import ScenarioPreset, SeedObjective
from pyrit.models.scenario_suite_execution import LaunchScenarioSuiteRequest, ScenarioSuiteLaunchEntrySnapshot
from pyrit.registry import (
    AttackTechniqueRegistry,
    InitializerRegistry,
    Registry,
    ScenarioPresetRegistry,
    ScenarioRegistry,
    ScenarioSuiteLaunchResolver,
    ScenarioSuiteRegistry,
)
from pyrit.scenario.core.dataset_overrides import apply_dataset_overrides
from pyrit.setup.initialization import reset_setup_registries
from pyrit.setup.initializers.scenario_suites import HorizontalTechniquesSuiteInitializer
from pyrit.setup.initializers.techniques import build_technique_factories

SUITE_NAME = HorizontalTechniquesSuiteInitializer.SUITE_NAME
LOCAL_DATASET_PATH = Path(__file__).parents[3] / "pyrit" / "datasets" / "seed_datasets" / "local"


def _configure_storage(tmp_path: Path) -> None:
    ScenarioPresetRegistry.get_registry_singleton().configure_source(str(tmp_path / "presets"))
    ScenarioSuiteRegistry.get_registry_singleton().configure_source(str(tmp_path / "suites"))


def _registered_presets() -> list[ScenarioPreset]:
    suite_entry = ScenarioSuiteRegistry.get_registry_singleton().get_suite(name=SUITE_NAME)
    assert suite_entry is not None
    preset_registry = ScenarioPresetRegistry.get_registry_singleton()
    presets = [preset_registry.instances.get(entry.preset_name) for entry in suite_entry.suite.entries]
    assert all(preset is not None for preset in presets)
    return presets  # type: ignore[return-value]


@pytest.fixture
def isolated_registries(tmp_path: Path) -> Iterator[None]:
    with patch.dict(Registry._singletons, {}, clear=True):
        _configure_storage(tmp_path)
        yield


@pytest.fixture
async def registered_suite_async(isolated_registries: None) -> None:
    await HorizontalTechniquesSuiteInitializer().initialize_async()


@pytest.fixture
def scenario_runtime(patch_central_database: object) -> Iterator[None]:
    with patch.dict(
        "os.environ",
        {
            "OPENAI_CHAT_ENDPOINT": "https://test.openai.azure.com/",
            "OPENAI_CHAT_KEY": "test-key",
            "OPENAI_CHAT_MODEL": "gpt-4",
        },
    ):
        AttackTechniqueRegistry.get_registry_singleton().register_from_factories(build_technique_factories())
        yield


@pytest.mark.usefixtures("registered_suite_async")
def test_initialize_registers_read_only_suite_with_presets_in_family_order() -> None:
    suite_entry = ScenarioSuiteRegistry.get_registry_singleton().get_suite(name=SUITE_NAME)

    assert suite_entry is not None
    assert suite_entry.origin == "registered"
    assert [entry.preset_name for entry in suite_entry.suite.entries] == [
        f"{SUITE_NAME}_jailbreaks",
        f"{SUITE_NAME}_multilingual",
        f"{SUITE_NAME}_encoding",
        f"{SUITE_NAME}_obfuscation",
    ]
    preset_registry = ScenarioPresetRegistry.get_registry_singleton()
    assert {entry.origin for entry in preset_registry.list_presets()} == {"registered"}


@pytest.mark.usefixtures("registered_suite_async")
def test_entry_ids_derive_from_suite_and_preset_names() -> None:
    suite_entry = ScenarioSuiteRegistry.get_registry_singleton().get_suite(name=SUITE_NAME)

    assert suite_entry is not None
    assert [entry.entry_id for entry in suite_entry.suite.entries] == [
        uuid5(NAMESPACE_URL, f"pyrit:scenario_suite:{SUITE_NAME}:{entry.preset_name}")
        for entry in suite_entry.suite.entries
    ]


@pytest.mark.usefixtures("registered_suite_async")
async def test_initialize_again_keeps_identical_definitions() -> None:
    await HorizontalTechniquesSuiteInitializer().initialize_async()

    assert len(ScenarioSuiteRegistry.get_registry_singleton().list_suites()) == 1
    assert len(ScenarioPresetRegistry.get_registry_singleton().list_presets()) == 4


@pytest.mark.usefixtures("registered_suite_async")
async def test_initialize_after_setup_registry_reset_reregisters_suite(tmp_path: Path) -> None:
    reset_setup_registries()
    _configure_storage(tmp_path)

    await HorizontalTechniquesSuiteInitializer().initialize_async()

    assert ScenarioSuiteRegistry.get_registry_singleton().get_suite(name=SUITE_NAME) is not None


@pytest.mark.usefixtures("isolated_registries")
async def test_initialize_rejects_a_different_preset_registered_under_a_suite_preset_name() -> None:
    ScenarioPresetRegistry.get_registry_singleton().register_preset(
        preset=ScenarioPreset(name=f"{SUITE_NAME}_encoding", scenario_name="foundry.red_team_agent")
    )

    with pytest.raises(ValueError, match=f"'{SUITE_NAME}_encoding' is already registered"):
        await HorizontalTechniquesSuiteInitializer().initialize_async()


@pytest.mark.usefixtures("registered_suite_async")
def test_every_preset_runs_every_shared_objective() -> None:
    datasets = [preset.datasets for preset in _registered_presets()]

    assert all(request == datasets[0] for request in datasets)
    assert datasets[0] is not None
    assert [source.name for source in datasets[0].sources or []] == ["airt_horizontal_objectives"]
    assert datasets[0].max_per_dataset == "all"
    assert datasets[0].max_total == "all"


@pytest.mark.usefixtures("registered_suite_async")
def test_technique_families_do_not_overlap() -> None:
    presets = _registered_presets()

    for first, second in itertools.combinations(presets, 2):
        if first.scenario_name == second.scenario_name:
            assert not set(first.techniques or []) & set(second.techniques or [])


@pytest.mark.usefixtures("registered_suite_async")
def test_only_the_jailbreak_family_runs_the_baseline_and_no_preset_is_listed_individually() -> None:
    presets = _registered_presets()

    assert [preset.name for preset in presets if preset.include_baseline] == [f"{SUITE_NAME}_jailbreaks"]
    assert not any(preset.show_in_scanner for preset in presets)


@pytest.mark.usefixtures("registered_suite_async", "scenario_runtime")
def test_every_preset_resolves_against_its_registered_scenario() -> None:
    preset_registry = ScenarioPresetRegistry.get_registry_singleton()
    scenario_registry = ScenarioRegistry.get_registry_singleton()

    for preset in _registered_presets():
        assert preset_registry.check_preset(preset=preset) == []
        scenario = scenario_registry.get_class(preset.scenario_name)()
        assert preset.datasets is not None
        effective = apply_dataset_overrides(
            config=scenario._default_dataset_config,
            request=preset.datasets,
            options=scenario.get_dataset_options(),
        )
        assert effective.dataset_names == ["airt_horizontal_objectives"]


@pytest.mark.usefixtures("registered_suite_async", "scenario_runtime")
def test_suite_launch_resolves_one_ordinary_run_per_preset() -> None:
    suite_entry = ScenarioSuiteRegistry.get_registry_singleton().get_suite(name=SUITE_NAME)
    assert suite_entry is not None
    request = LaunchScenarioSuiteRequest(
        target_name="objective_target",
        entries=[
            ScenarioSuiteLaunchEntrySnapshot(
                entry_id=entry.entry_id, preset_name=entry.preset_name, origin="registered", preset_version=None
            )
            for entry in suite_entry.suite.entries
        ],
    )

    resolved = ScenarioSuiteLaunchResolver().resolve(name=SUITE_NAME, request=request)

    assert [entry.request.scenario_name for entry in resolved.entries] == [
        "airt.jailbreak",
        "airt.multilingual",
        "foundry.red_team_agent",
        "foundry.red_team_agent",
    ]


def test_suite_initializer_is_discovered_and_the_abstract_base_is_not() -> None:
    names = InitializerRegistry(lazy_discovery=False).get_class_names()

    assert "horizontal_techniques_suite" in names
    assert "scenario_suite" not in names


async def test_shared_objective_dataset_is_one_objective_per_attack() -> None:
    loader = _LocalDatasetLoader(file_path=LOCAL_DATASET_PATH / "airt" / "horizontal_objectives.prompt")

    dataset = await loader.fetch_dataset_async()

    assert dataset.dataset_name in HorizontalTechniquesSuiteInitializer.DATASET_NAMES
    assert dataset.seeds
    assert all(isinstance(seed, SeedObjective) and seed.harm_categories for seed in dataset.seeds)
    assert len(dataset.seed_groups) == len(dataset.seeds)
