# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Scenario Presets and Suites
#
# A **scenario preset** is a named, target-agnostic scenario configuration: a registered scenario
# plus the techniques, datasets, baseline choice, and scenario parameters to run it with. The target,
# concurrency, retries, and labels are chosen when the preset is launched, so one preset runs
# unchanged against any target.
#
# A **scenario suite** is a named, ordered list of preset references. Launching a suite against a
# target runs each referenced preset as an ordinary scenario run, in order. Each child run records
# which suite launch and preset produced it, so the results of one launch can be compared side by side.
#
# Presets and suites come from two places:
#
# - **Stored** definitions are JSON documents that operators create and edit on the GUI's
#   *Scenario presets* registry page. Their locations are set by `scenario_presets_source` and
#   `scenario_suites_source` (see [Configuration](../../getting_started/pyrit_conf.md)).
# - **Registered** definitions are contributed by code, usually by an initializer, and are read-only.
#   A stored document with the same name takes precedence.
#
# ## How a suite launch works
#
# The GUI Scanner lists every suite in its *Scenario suites* table. Launching one sends
# `POST /api/scenario-suites/{name}/launch` with the target and the suite entries the operator
# reviewed. The backend then:
#
# 1. Resolves the suite and each referenced preset, and rejects the launch if any of them changed
#    since it was reviewed or no longer resolves against the registered scenarios.
# 2. Turns each entry into an ordinary scenario run request by adding the launch's target and
#    execution settings to the preset.
# 3. Submits the runs in suite order, stamping each result's metadata with a
#    `scenario_suite_execution` entry that names the suite, the entry, and the preset.
# 4. Tracks the runs together on the Scanner's suite progress page, where they can be cancelled as a
#    group.
#
# This notebook performs the same resolution in Python and runs the resulting scenarios directly.

# %% [markdown]
# ## The built-in horizontal techniques suite
#
# The `horizontal_techniques_suite` initializer registers the `horizontal_techniques` suite. It runs
# one preset per attack technique family — jailbreak templates, multilingual translation, encodings,
# and character obfuscation — against the same objectives, so per-family success rates are directly
# comparable. Each family runs a small representative subset of its techniques, and the baseline runs
# once, with the jailbreak preset. To offer the suite in the GUI, add the initializer to your
# configuration file:
#
# ```yaml
# initializers:
#   - name: target
#   - name: scorer
#   - name: technique
#   - name: horizontal_techniques_suite
# ```
#
# Here the initializer runs directly after the shared documentation configuration.

# %%
from pathlib import Path

from pyrit.registry import ScenarioPresetRegistry, ScenarioSuiteRegistry, TargetRegistry
from pyrit.setup import initialize_from_config_async
from pyrit.setup.initializers import HorizontalTechniquesSuiteInitializer

await initialize_from_config_async(config_path=Path("../../scanner/pyrit_conf.yaml"))  # type: ignore
await HorizontalTechniquesSuiteInitializer().initialize_async()  # type: ignore

objective_target = TargetRegistry.get_registry_singleton().instances.get("openai_chat")
suite_registry = ScenarioSuiteRegistry.get_registry_singleton()
preset_registry = ScenarioPresetRegistry.get_registry_singleton()

# %% [markdown]
# The suite stores only ordered preset names. Each preset holds the scenario configuration, and every
# preset selects the same dataset without a size limit.

# %%
suite = suite_registry.get_suite(name="horizontal_techniques")
print(f"{suite.suite.name} ({suite.origin}): {suite.suite.description}\n")

for position, entry in enumerate(suite.suite.entries, start=1):
    preset = preset_registry.get_preset(name=entry.preset_name).preset
    print(f"{position}. {preset.name} -> {preset.scenario_name}")
    print(f"   techniques: {preset.techniques}")
    print(f"   scenario parameters: {preset.scenario_params}")
    print(f"   baseline: {preset.include_baseline}")
    print(f"   datasets: {[source.name for source in preset.datasets.sources]}, max_total={preset.datasets.max_total}")

# %% [markdown]
# ## Resolve a launch
#
# A launch names the target and repeats the suite entries the operator reviewed, with each preset's
# origin and version. `ScenarioSuiteLaunchResolver` checks that snapshot against the current
# definitions, validates every preset, and returns one ordinary scenario run request per entry.

# %%
from pyrit.models import LaunchScenarioSuiteRequest, ScenarioSuiteLaunchEntrySnapshot
from pyrit.registry import ScenarioSuiteLaunchResolver

reviewed_entries = []
for entry in suite.suite.entries:
    preset_entry = preset_registry.get_preset(name=entry.preset_name)
    reviewed_entries.append(
        ScenarioSuiteLaunchEntrySnapshot(
            entry_id=entry.entry_id,
            preset_name=entry.preset_name,
            origin=preset_entry.origin,
            preset_version=preset_entry.version,
        )
    )

launch_request = LaunchScenarioSuiteRequest(
    target_name="openai_chat",
    expected_version=suite.version,
    entries=reviewed_entries,
)
resolved = ScenarioSuiteLaunchResolver().resolve(name=suite.suite.name, request=launch_request)

for entry in resolved.entries:
    request = entry.request
    print(f"{entry.entry_position}. {entry.preset_name}: {request.scenario_name} against {request.target_name}")

# %% [markdown]
# ## Run the suite
#
# The cell below builds each scenario through the scenario registry from its resolved request, attaches
# the suite attribution metadata, and runs the scenarios in suite order.
#
# To keep this example fast, every child is narrowed to the single `scams` objective with a dataset
# filter, so each preset still runs the same objective. A launch from the GUI runs every objective in
# the dataset.

# %%
from uuid import uuid4

from pyrit.registry import ScenarioRegistry
from pyrit.scenario.core import resolve_technique_tokens

scenario_registry = ScenarioRegistry.get_registry_singleton()
suite_execution_id = uuid4()
results = []

for entry in resolved.entries:
    request = entry.request
    technique_class = scenario_registry.get_registered_class_metadata(request.scenario_name).technique_class
    techniques = resolve_technique_tokens(
        tokens=request.techniques,
        technique_class=technique_class,
        scenario_name=request.scenario_name,
    )
    scenario = await scenario_registry.create_and_initialize_async(  # type: ignore
        request.scenario_name,
        scenario_params=request.scenario_params,
        initial_metadata=entry.metadata(suite_execution_id=suite_execution_id, suite=resolved.suite),
        objective_target=objective_target,
        scenario_techniques=techniques.techniques,
        dataset_overrides=request.datasets.model_copy(update={"filters": {"harm_categories": ["scams"]}}),
        include_baseline=request.include_baseline,
    )
    results.append(await scenario.run_async())  # type: ignore

# %% [markdown]
# ## Compare technique families
#
# Each child result carries its attribution under `metadata["scenario_suite_execution"]`.
# `compute_scenario_statistics` counts every execution unit once, by its latest attempt, and groups
# the units by the scenario's display groups: jailbreak templates, languages, or converter techniques.
# The jailbreak preset's total includes its baseline. A refusal counts as a failure; an error means
# the attack could not complete.

# %%
from pyrit.analytics import compute_scenario_statistics
from pyrit.models import SCENARIO_SUITE_EXECUTION_METADATA_KEY, ScenarioProgressCounts


def summarize(counts: ScenarioProgressCounts) -> str:
    outcomes = counts.outcomes
    return f"{outcomes.successes} succeeded, {outcomes.failures} failed, {outcomes.errors} errored"


for result in results:
    attribution = result.metadata[SCENARIO_SUITE_EXECUTION_METADATA_KEY]
    statistics = compute_scenario_statistics(result)
    print(f"{attribution['entry_position']}. {attribution['preset_name']}: {summarize(statistics.overall)}")
    for group, counts in statistics.display_groups.items():
        print(f"     {group}: {summarize(counts)}")

# %% [markdown]
# ## Define your own suite
#
# Subclass `ScenarioSuiteInitializer`, name the suite, and return its presets in launch order. Entry
# identifiers are derived from the suite and preset names, so they stay the same across processes, and
# running the initializer again leaves identical definitions in place. Hide presets that only exist
# for the suite with `show_in_scanner=False`.
#
# The example below runs one role-play technique against two harm areas. To load a suite like this in
# a deployment, save the class in a script listed under `initialization_scripts` in your configuration
# file. Suites can also be built without code on the GUI's *Scenario presets* registry page.

# %%
from typing import ClassVar

from pyrit.models import DatasetSourceInput, ScenarioDatasetRequest, ScenarioPreset
from pyrit.setup.initializers import ScenarioSuiteInitializer


class HarmSpotCheckSuiteInitializer(ScenarioSuiteInitializer):
    """Register a suite that runs one role-play technique against two harm areas."""

    SUITE_NAME: ClassVar[str] = "harm_spot_check"
    SUITE_DESCRIPTION: ClassVar[str] = "Runs the movie-script role play against hate and violence objectives."

    def _build_presets(self) -> list[ScenarioPreset]:
        return [
            ScenarioPreset(
                name=f"{self.SUITE_NAME}_{dataset_name}",
                scenario_name="airt.rapid_response",
                show_in_scanner=False,
                techniques=["role_play_movie_script"],
                datasets=ScenarioDatasetRequest(sources=[DatasetSourceInput(name=dataset_name)], max_total=2),
            )
            for dataset_name in ("airt_hate", "airt_violence")
        ]


await HarmSpotCheckSuiteInitializer().initialize_async()  # type: ignore

custom_suite = suite_registry.get_suite(name="harm_spot_check")
for entry in custom_suite.suite.entries:
    preset = preset_registry.get_preset(name=entry.preset_name).preset
    print(f"{entry.entry_id} {preset.name}: issues={preset_registry.check_preset(preset=preset)}")
