# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Base initializer for registering one code-owned scenario suite and the presets it runs."""

import asyncio
import logging
from abc import abstractmethod
from typing import ClassVar
from uuid import NAMESPACE_URL, UUID, uuid5

from pyrit.models import ScenarioPreset, ScenarioSuite, ScenarioSuiteEntry
from pyrit.registry import ScenarioPresetRegistry, ScenarioSuiteRegistry
from pyrit.setup.pyrit_initializer import PyRITInitializer

logger = logging.getLogger(__name__)


class ScenarioSuiteInitializer(PyRITInitializer):
    """
    Register one read-only scenario suite together with the presets it runs.

    Subclasses name the suite and build its presets in launch order. Each preset appears
    once, and its suite entry identifier is derived from the suite and preset names, so it
    stays stable across processes and deployments.

    Registration is idempotent: running the initializer again in the same process leaves
    identical definitions in place. A different definition already registered under one
    of the names is reported rather than silently replaced.
    """

    SUITE_NAME: ClassVar[str]
    SUITE_DESCRIPTION: ClassVar[str]

    async def initialize_async(self) -> None:
        """Build the suite's presets and register them with the suite."""
        presets = self._build_presets()
        suite = self._build_suite(presets=presets)
        # Registration checks storage for shadowing documents, so keep it off the event loop.
        await asyncio.to_thread(self._register, presets=presets, suite=suite)
        logger.info("Registered scenario suite '%s' with %d preset(s).", suite.name, len(presets))

    @abstractmethod
    def _build_presets(self) -> list[ScenarioPreset]:
        """
        Build the presets this suite runs, in launch order.

        Returns:
            list[ScenarioPreset]: The suite's presets, each listed once.
        """

    def _build_suite(self, *, presets: list[ScenarioPreset]) -> ScenarioSuite:
        """
        Build the suite referencing each preset in order.

        Args:
            presets (list[ScenarioPreset]): The suite's presets in launch order.

        Returns:
            ScenarioSuite: The suite definition.
        """
        return ScenarioSuite(
            name=self.SUITE_NAME,
            description=self.SUITE_DESCRIPTION,
            entries=[
                ScenarioSuiteEntry(entry_id=self._entry_id(preset_name=preset.name), preset_name=preset.name)
                for preset in presets
            ],
        )

    def _entry_id(self, *, preset_name: str) -> UUID:
        """
        Derive the stable suite entry identifier for one preset.

        Args:
            preset_name (str): The referenced preset name.

        Returns:
            UUID: An identifier that is the same in every process for this suite and preset.
        """
        return uuid5(NAMESPACE_URL, f"pyrit:scenario_suite:{self.SUITE_NAME}:{preset_name}")

    def _register(self, *, presets: list[ScenarioPreset], suite: ScenarioSuite) -> None:
        """
        Register presets before the suite that references them.

        Args:
            presets (list[ScenarioPreset]): The suite's presets.
            suite (ScenarioSuite): The suite definition.
        """
        tags = {"scenario_suite": suite.name}

        preset_registry = ScenarioPresetRegistry.get_registry_singleton()
        for preset in presets:
            if not _is_already_registered(registered=preset_registry.instances.get(preset.name), definition=preset):
                preset_registry.register_preset(preset=preset, tags=tags)

        suite_registry = ScenarioSuiteRegistry.get_registry_singleton()
        if not _is_already_registered(registered=suite_registry.instances.get(suite.name), definition=suite):
            suite_registry.register_suite(suite=suite, tags=tags)


def _is_already_registered(
    *,
    registered: ScenarioPreset | ScenarioSuite | None,
    definition: ScenarioPreset | ScenarioSuite,
) -> bool:
    """
    Report whether an identical definition is already registered under the same name.

    Args:
        registered (ScenarioPreset | ScenarioSuite | None): The definition currently registered
            under ``definition.name``, if any.
        definition (ScenarioPreset | ScenarioSuite): The definition about to be registered.

    Returns:
        bool: True if the identical definition is already registered, False if the name is free.

    Raises:
        ValueError: If a different definition is already registered under the name.
    """
    if registered is None:
        return False
    if registered != definition:
        raise ValueError(
            f"'{definition.name}' is already registered with a different definition; "
            f"choose a different suite or preset name."
        )
    return True
