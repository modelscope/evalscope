"""Typed configuration and dispatch for external performance scenarios."""

import json
from typing import TYPE_CHECKING, Annotated, Any, Dict, Union

from pydantic import Field, TypeAdapter

from .agentx import AgentXScenario, run_agentx_benchmark, validate_agentx_arguments
from .agentx_harness import AgentXHarnessScenario, run_agentx_harness_benchmark, validate_agentx_harness_arguments

if TYPE_CHECKING:
    from evalscope.perf.arguments import Arguments

PerfScenario = Annotated[Union[AgentXScenario, AgentXHarnessScenario], Field(discriminator='name')]
_SCENARIO_ADAPTER = TypeAdapter(PerfScenario)

__all__ = [
    'AgentXScenario',
    'AgentXHarnessScenario',
    'PerfScenario',
    'parse_scenario',
    'run_agentx_benchmark',
    'run_agentx_harness_benchmark',
    'run_scenario_benchmark',
    'validate_scenario_arguments',
]


def parse_scenario(value: Any) -> PerfScenario:
    """Parse scenario shorthand, JSON, mappings, or an existing typed model."""
    if isinstance(value, str):
        if value in {'agentx', 'agentx_harness', 'agentx_harness_256k'}:
            value = {'name': value}
        else:
            try:
                value = json.loads(value)
            except ValueError as error:
                raise ValueError(
                    '--scenario must be agentx, agentx_harness, agentx_harness_256k, or a scenario JSON object.'
                ) from error
    if isinstance(value, dict):
        # Preserve legacy JSON configurations that omitted the scenario name.
        value = {'name': 'agentx', **value}
    return _SCENARIO_ADAPTER.validate_python(value)


def validate_scenario_arguments(args: 'Arguments') -> None:
    """Validate Perf controls against the explicitly selected scenario."""
    if isinstance(args.scenario, AgentXHarnessScenario):
        validate_agentx_harness_arguments(args)
    else:
        validate_agentx_arguments(args)


def run_scenario_benchmark(args: 'Arguments', output_path: str) -> Dict[str, Dict[str, Any]]:
    """Dispatch without detecting installed engines or changing methodologies."""
    if isinstance(args.scenario, AgentXHarnessScenario):
        return run_agentx_harness_benchmark(args, output_path)
    if isinstance(args.scenario, AgentXScenario):
        return run_agentx_benchmark(args, output_path)
    raise ValueError('An external performance scenario is required.')
