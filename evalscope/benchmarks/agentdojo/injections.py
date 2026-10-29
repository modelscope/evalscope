"""Injection planning: model-name resolution and per-sample ``{injection_vector: text}`` maps.

Injection texts come from the upstream attack classes, reached through the ``UpstreamContext`` handed in by the
adapter and through imports inside function bodies only.
"""

import functools
import random
from typing import Any, Dict, Mapping, Optional, Tuple

from evalscope.benchmarks.agentdojo.constants import BENIGN_ATTACK, DEFAULT_MODEL_NAME
from evalscope.benchmarks.agentdojo.upstream import UpstreamContext, UpstreamDataError

# Attacks whose ``{model}`` placeholder always uses the upstream default model name.
_DEFAULT_MODEL_NAME_ATTACKS = ('important_instructions_no_model_name', 'important_instructions_no_names')
_WRONG_MODEL_NAME_ATTACK = 'important_instructions_wrong_model_name'


def resolve_victim_model_name(
    model_id: str, override: Optional[str], model_names: Mapping[str, str]
) -> Tuple[str, bool]:
    """Return ``(victim_model_name, used_default)``.

    A non-empty ``override`` wins. Otherwise the first ``model_names`` key (in iteration order) that is a
    case-sensitive substring of ``model_id`` selects its name, as upstream does; otherwise the upstream default name
    is returned with ``used_default=True``.
    """
    if override:
        return override, False
    for key, name in model_names.items():
        if key in model_id:
            return name, False
    return DEFAULT_MODEL_NAME, True


def resolve_effective_model_name(attack_name: str, victim: str, model_names: Mapping[str, str]) -> str:
    """Return the name substituted for ``{model}`` in the attack template.

    ``*_no_model_name`` / ``*_no_names`` use the upstream default name, ``*_wrong_model_name`` mirrors upstream
    ``get_wrong_model_name_from_pipeline`` for the resolved victim, and every other attack uses ``victim``.
    """
    if attack_name in _DEFAULT_MODEL_NAME_ATTACKS:
        return DEFAULT_MODEL_NAME
    if attack_name == _WRONG_MODEL_NAME_ATTACK:
        return random.Random(0).choice([name for name in model_names.values() if name != victim])
    return victim


@functools.lru_cache(maxsize=1)
def _pipeline_stub_class() -> type:
    """Build the pipeline stub lazily, because it must subclass an upstream type."""
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement

    class _PipelineStub(BasePipelineElement):
        """Pipeline that only carries a name; attack constructors read ``target_pipeline.name``."""

        def __init__(self, name: str) -> None:
            self.name = name

        def query(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError('The AgentDojo pipeline stub cannot run queries.')

    return _PipelineStub


class InjectionPlanner:
    """Build per-sample injection maps with the upstream attack classes.

    One attack instance is built per suite. Upstream caches injection candidates per ``(attack instance, user
    task)``, so they are computed once per user task and shared by every injection task and repeat.
    """

    def __init__(self, upstream: UpstreamContext, attack_name: str, victim_model_name: str) -> None:
        self._upstream = upstream
        self.attack_name = attack_name
        self.victim_model_name = victim_model_name
        self.effective_model_name = resolve_effective_model_name(attack_name, victim_model_name, upstream.model_names)
        self._attacks: Dict[str, Any] = {}
        if attack_name == BENIGN_ATTACK:
            return
        # Model-name attacks resolve a name from the pipeline in their constructor, so the stub carries a known
        # MODEL_NAMES key; the resolved name is then replaced by the effective model name.
        stub = _pipeline_stub_class()(next(iter(upstream.model_names)))
        for suite_name, suite in upstream.suites.items():
            attack = upstream.attacks[attack_name](suite, stub)
            attack.model_name = self.effective_model_name
            self._attacks[suite_name] = attack

    def injections_for(self, suite_name: str, user_task_id: str, injection_task_id: Optional[str]) -> Dict[str, str]:
        """Return the upstream ``attack(user_task, injection_task)`` map; ``{}`` for benign runs.

        Raises:
            UpstreamDataError: The user task has no injection candidates, or a ``tool_knowledge`` injection task
                lacks ``placeholder_args``.
        """
        if self.attack_name == BENIGN_ATTACK:
            return {}
        suite = self._upstream.suites[suite_name]
        try:
            return dict(
                self._attacks[suite_name].attack(
                    suite.user_tasks[user_task_id], suite.injection_tasks[injection_task_id]
                )
            )
        except ValueError as exc:
            if 'is not injectable' in str(exc):
                reason = f'user task {user_task_id!r} has no injection candidates'
            elif 'must contain a placeholder' in str(exc):
                reason = f'the ground-truth tool calls of injection task {injection_task_id!r} lack placeholder_args'
            else:
                raise
            raise UpstreamDataError(
                f'AgentDojo attack {self.attack_name!r} cannot build injections for suite {suite_name!r} '
                f'({user_task_id!r}, {injection_task_id!r}): {reason} (upstream: {exc}).'
            ) from exc
