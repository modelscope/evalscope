"""AgentAdapter: first-class base class for Agent benchmarks.

Builds on :class:`DefaultDataAdapter` by taking over ``_on_inference`` and
wiring together a :class:`Strategy`, :class:`ToolExecutor` and optional
:class:`AgentEnvironment` into an :class:`AgentLoop` per sample.

Subclasses typically only override :meth:`build_strategy`,
:meth:`build_tools` and :meth:`build_environment`.  Global
``TaskConfig.agent_config`` is intentionally **ignored** here - benchmarks
deriving from this class already define their own Agent harness.
"""

from typing import Any, Dict, List, Optional

from evalscope.api.agent import (
    AgentContext,
    AgentEnvironment,
    AgentLoop,
    AgentStrategy,
    AgentTrace,
    ToolExecutor,
    ToolHandler,
)
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import get_strategy
from evalscope.utils.function_utils import AsyncioLoopRunner
from .default_data_adapter import DefaultDataAdapter


class AgentAdapter(DefaultDataAdapter):
    """Adapter for benchmarks that require a multi-turn Agent loop.

    Concrete benchmarks (e.g. ``swe_bench_pro``) subclass this and override
    the ``build_*`` hooks.  The synchronous ``_on_inference`` bridges into
    the async :class:`AgentLoop` via :class:`AsyncioLoopRunner`.
    """

    #: Default strategy name used when :meth:`build_strategy` is not
    #: overridden.  Subclasses can change this (e.g.
    #: ``'swe_bench_toolcall'``).
    strategy_name: str = 'function_calling'

    #: Upper bound on loop iterations per sample.
    max_steps: int = 30

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

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

        Default: no tools.  Override to pull from ``self._task_config`` or
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

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        """Drive :class:`AgentLoop` for this sample and return the final output.

        Ignores ``TaskConfig.agent_config`` entirely; native Agent
        benchmarks are self-contained by design.
        """
        strategy = self.build_strategy(sample)
        handlers = self.build_tools(sample)
        environment = self.build_environment(sample)

        trace = AgentTrace(
            strategy=getattr(strategy, 'name', None),
            environment=environment.name if environment else None,
            max_steps=self.max_steps,
        )

        async def _run() -> Any:
            tool_executor = ToolExecutor(handlers=handlers, environment=environment)
            ctx = AgentContext(
                sample_id=sample.id,
                messages=self.build_initial_messages(sample),
                tools=list(sample.tools or []),
                max_steps=self.max_steps,
            )
            loop = AgentLoop(
                model=model,
                strategy=strategy,
                tool_executor=tool_executor,
                environment=environment,
                max_steps=self.max_steps,
                trace=trace,
            )
            try:
                return await loop.run(ctx)
            finally:
                if environment is not None:
                    await environment.close()

        result = AsyncioLoopRunner.run(_run())

        # Resolve the final prediction through the strategy → adapter hook
        # chain so benchmarks (e.g. SWE-bench) can extract a custom payload
        # like a git patch from the trajectory.
        final_text = self._extract_final_answer(result, strategy)

        output = result.final_output
        output.completion = final_text
        if output.metadata is None:
            output.metadata = {}
        output.metadata['__agent_messages__'] = result.messages
        output.metadata['__agent_trace__'] = result.trace
        return output

    def _extract_final_answer(self, result: Any, strategy: AgentStrategy) -> str:
        """Override hook for adapters that need custom prediction extraction.

        Default implementation forwards to ``strategy.extract_final_answer``.
        SWE-bench agentic benchmarks override this to scan tool messages for
        the ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` sentinel and return
        the patch text following it.
        """
        return strategy.extract_final_answer(result)
