"""AgentLoopAdapter: AgentLoop-driven base class for agent benchmarks.

Builds on :class:`AgentAdapter` (a pure marker base) by taking over
``_on_inference`` and wiring together a :class:`AgentStrategy`,
:class:`ToolExecutor` and optional :class:`AgentEnvironment` into an
:class:`AgentLoop` per sample.

Subclasses typically only override :meth:`build_strategy`,
:meth:`build_tools` and :meth:`build_environment`. Global
``TaskConfig.agent_config`` is intentionally **ignored** here — benchmarks
deriving from this class already define their own Agent harness.
"""

from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment, AgentLoopResult, AgentStrategy, ToolHandler
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import Model
from evalscope.api.registry import get_strategy
from ._agent_loop_runner import run_agent_loop
from .agent_adapter import AgentAdapter


class AgentLoopAdapter(AgentAdapter):
    """Adapter for benchmarks that drive a generic multi-turn AgentLoop.

    Concrete benchmarks (e.g. ``swe_bench_*_agentic``) subclass this and
    override the ``build_*`` hooks. The synchronous :meth:`_on_inference`
    bridges into the async :class:`AgentLoop` via the shared
    :func:`run_agent_loop` helper.
    """

    #: Default strategy name used when :meth:`build_strategy` is not
    #: overridden. Subclasses can change this (e.g.
    #: ``'swe_bench_toolcall'``).
    strategy_name: str = 'function_calling'

    #: Default upper bound on loop iterations per sample. Subclasses may
    #: override either via class attribute or by pushing a ``max_steps``
    #: entry into ``extra_params`` (read in ``__init__``).
    max_steps_default: int = 30

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Allow benchmarks to expose ``max_steps`` to end users via the
        # ``extra_params`` block of their ``BenchmarkMeta`` registration.
        self.max_steps = int(self.extra_params.get('max_steps', self.max_steps_default))

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def build_strategy(self, sample: Sample) -> AgentStrategy:
        """Return the :class:`AgentStrategy` instance for this sample.

        Default: lookup ``self.strategy_name`` in the strategy registry and
        instantiate with default parameters.
        """
        strategy_cls = get_strategy(self.strategy_name)
        return strategy_cls()

    def build_tools(self, sample: Sample) -> Dict[str, ToolHandler]:
        """Return the ``{tool_name: handler}`` map for the per-sample loop.

        Default: no tools. Override to pull from ``self._task_config`` or
        per-sample metadata.
        """
        return {}

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        """Return an :class:`AgentEnvironment` or ``None`` if not needed."""
        return None

    def build_initial_messages(self, sample: Sample) -> List[Any]:
        """Return the message list the loop starts with.

        Default: promote ``sample.input`` into a single
        :class:`ChatMessageUser` when it is a plain string, otherwise copy
        the list.
        """
        if isinstance(sample.input, list):
            return list(sample.input)
        return [ChatMessageUser(content=sample.input)]

    # ------------------------------------------------------------------
    # Overridden inference hook
    # ------------------------------------------------------------------

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        """Drive :class:`AgentLoop` for this sample and return the final output.

        Ignores ``TaskConfig.agent_config`` entirely; native AgentLoop
        benchmarks are self-contained by design.
        """
        strategy = self.build_strategy(sample)
        handlers = self.build_tools(sample)
        environment = self.build_environment(sample)

        result: AgentLoopResult = run_agent_loop(
            model=model,
            strategy=strategy,
            handlers=handlers,
            environment=environment,
            initial_messages=self.build_initial_messages(sample),
            all_tools=list(sample.tools or []),
            max_steps=self.max_steps,
            sample_id=sample.id,
            trace_strategy_name=getattr(strategy, 'name', None),
            trace_env_name=environment.name if environment else None,
        )

        # Resolve the final prediction through the strategy → adapter hook
        # chain so benchmarks (e.g. SWE-bench) can extract a custom payload
        # like a git patch from the trajectory.
        final_text = self._extract_final_answer(result, strategy)

        output = result.final_output
        output.completion = final_text
        return InferenceResult(output=output, messages=result.messages, trace=result.trace)

    def _extract_final_answer(self, result: AgentLoopResult, strategy: AgentStrategy) -> str:
        """Override hook for adapters that need custom prediction extraction.

        Default implementation forwards to ``strategy.extract_final_answer``.
        SWE-bench agentic benchmarks override this to scan tool messages for
        the ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` sentinel and return
        the patch text following it.
        """
        return strategy.extract_final_answer(result)
