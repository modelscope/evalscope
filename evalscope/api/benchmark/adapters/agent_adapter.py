"""AgentAdapter & AgentLoopAdapter: base classes for agent-class benchmarks.

:class:`AgentAdapter` is a structural marker that classifies a benchmark as
belonging to the *agent* family — used by documentation generators (see
``generate_dataset_md.py`` which calls
``issubclass(adapter_cls, AgentAdapter)``) and downstream tooling.

Two extension modes are supported:

1. **Custom multi-turn loop** (e.g. ``tau_bench``, ``bfcl_v3/v4``,
   ``general_fc``): subclass :class:`AgentAdapter` directly and override
   :meth:`_on_inference` to drive the benchmark's bespoke loop. The
   built-in :class:`evalscope.api.agent.AgentLoop` is **not** used.

2. **Generic AgentLoop driving** (e.g. ``swe_bench_*_agentic``): subclass
   :class:`AgentLoopAdapter` instead, which derives from this class and
   wires together :class:`AgentLoop` + ``build_*`` hooks.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .default_data_adapter import DefaultDataAdapter

if TYPE_CHECKING:
    from evalscope.agent.external.runners import AgentRunResult
    from evalscope.api.agent.mcp import MCPServerConfig


class AgentAdapter(DefaultDataAdapter):
    """Marker base class for agent-class benchmarks.

    Subclasses are free to override :meth:`_on_inference` to plug in their
    own multi-turn driver. For benchmarks that want the standard
    :class:`AgentLoop` orchestration, subclass :class:`AgentLoopAdapter`
    instead.
    """

    pass


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

    #: Default upper bound on loop iterations per sample. Subclasses override
    #: this class attribute; users can explicitly override it through
    #: ``NativeAgentConfig.max_steps``.
    max_steps_default: int = 30

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_steps = self.max_steps_default

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def build_strategy(self, sample: Any) -> Any:
        """Return the :class:`AgentStrategy` instance for this sample.

        Default: lookup ``self.strategy_name`` in the strategy registry and
        instantiate with default parameters.
        """
        from evalscope.api.registry import get_strategy

        strategy_cls = get_strategy(self.strategy_name)
        return strategy_cls()

    def build_tools(self, sample: Any) -> Dict[str, Any]:
        """Return the ``{tool_name: handler}`` map for the per-sample loop.

        Default: no tools. Override to pull from ``self._task_config`` or
        per-sample metadata.
        """
        return {}

    def build_environment(self, sample: Any) -> Optional[Any]:
        """Return an :class:`AgentEnvironment` or ``None`` if not needed."""
        return None

    def _task_sandbox_config(self) -> Dict[str, Any]:
        """Return task-level sandbox defaults for benchmark-owned environments."""
        if self._task_config is None or self._task_config.sandbox is None:
            return {}
        return dict(self._task_config.sandbox.default_config or {})

    def build_initial_messages(self, sample: Any) -> List[Any]:
        """Return the message list the loop starts with.

        Default: promote ``sample.input`` into a single
        :class:`ChatMessageUser` when it is a plain string, otherwise copy
        the list.
        """
        from evalscope.api.messages import ChatMessageUser

        if isinstance(sample.input, list):
            return list(sample.input)
        return [ChatMessageUser(content=sample.input)]

    # ------------------------------------------------------------------
    # Overridden inference hook
    # ------------------------------------------------------------------

    def _on_inference(self, model: Any, sample: Any) -> Any:
        """Drive :class:`AgentLoop` for this sample and return the final output.

        Benchmark defaults remain authoritative when ``agent_config`` is
        omitted. Explicit ``NativeAgentConfig`` strategy, tools and max_steps
        fields override or extend those defaults; MCP tools are always merged.
        The benchmark keeps ownership of its environment so task-specific
        mounts and sandbox contracts remain intact.

        :class:`ExternalAgentConfig` routes through :func:`run_external_agent`
        directly, with the adapter's :meth:`build_environment` and
        :meth:`build_initial_messages` supplying the per-sample sandbox
        and prompt, and :meth:`_external_extract_prediction` recovering
        the prediction artifact before the env closes.
        """
        from evalscope.api.agent import AgentLoopResult, NativeAgentConfig, run_agent_loop
        from evalscope.api.evaluator import InferenceResult
        from evalscope.api.registry import get_strategy, resolve_tool_infos, resolve_tools

        ac = self._task_config.agent_config if self._task_config is not None else None
        if ac is not None:
            # Local import to keep the bridge stack out of the adapter's
            # module-load-time imports (no aiohttp dependency for non-
            # external benchmark runs).
            from evalscope.agent.external.adapter import run_external_agent
            from evalscope.agent.external.config import ExternalAgentConfig
            if isinstance(ac, ExternalAgentConfig):
                messages = self.build_initial_messages(sample)
                instruction = '\n\n'.join(m.text for m in messages if getattr(m, 'text', ''))
                return run_external_agent(
                    config=ac,
                    model=model,
                    sample=sample,
                    environment_override=self.build_environment(sample),
                    instruction_override=instruction,
                    post_run_hook=self._external_extract_prediction,
                )

        mcp_configs: Optional[List['MCPServerConfig']] = None
        strategy = self.build_strategy(sample)
        handlers = self.build_tools(sample)
        all_tools = list(sample.tools or [])
        max_steps = self.max_steps
        if isinstance(ac, NativeAgentConfig):
            # Do not let NativeAgentConfig's generic defaults replace benchmark
            # defaults when the user only configured tools or MCP servers.
            explicit_fields = ac.model_fields_set
            if 'strategy' in explicit_fields or 'kwargs' in explicit_fields:
                strategy_name = ac.strategy if 'strategy' in explicit_fields else strategy.name
                strategy = get_strategy(strategy_name)(**ac.kwargs)
            if 'max_steps' in explicit_fields:
                max_steps = ac.max_steps
            if ac.tools:
                configured_handlers = resolve_tools(ac.tools)
                # Benchmark handlers win name collisions so required task
                # semantics cannot be replaced by a global config.
                handlers = {**configured_handlers, **handlers}
                existing_tool_names = {tool.name for tool in all_tools}
                for tool_info in resolve_tool_infos(ac.tools):
                    if tool_info.name not in existing_tool_names:
                        all_tools.append(tool_info)
                        existing_tool_names.add(tool_info.name)
            mcp_configs = ac.mcp_servers or None

        if max_steps <= 0:
            raise ValueError('AgentLoop max_steps must be greater than 0.')
        environment = self.build_environment(sample)

        result: AgentLoopResult = run_agent_loop(
            model=model,
            strategy=strategy,
            handlers=handlers,
            environment=environment,
            initial_messages=self.build_initial_messages(sample),
            all_tools=all_tools,
            max_steps=max_steps,
            sample_id=sample.id,
            trace_strategy_name=getattr(strategy, 'name', None),
            trace_env_name=environment.name if environment else None,
            mcp_configs=mcp_configs,
        )

        # Resolve the final prediction through the strategy → adapter hook
        # chain so benchmarks (e.g. SWE-bench) can extract a custom payload
        # like a git patch from the trajectory.
        final_text = self._extract_final_answer(result, strategy)

        output = result.final_output
        output.completion = final_text
        return InferenceResult(output=output, messages=result.messages, trace=result.trace)

    async def _external_extract_prediction(
        self,
        env: Any,
        run_result: 'AgentRunResult',
        sample: Any,
    ) -> str:
        """Recover the prediction text from the sandbox after the
        external agent finishes.

        Default implementation forwards the runner's stdout. SWE-bench
        adapters override this to call
        :func:`evalscope.agent.external.helpers.extract_patch`.
        """
        return run_result.output

    def _extract_final_answer(self, result: Any, strategy: Any) -> str:
        """Override hook for adapters that need custom prediction extraction.

        Default implementation forwards to ``strategy.extract_final_answer``.
        SWE-bench agentic benchmarks override this to scan tool messages for
        the ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` sentinel and return
        the patch text following it.
        """
        return strategy.extract_final_answer(result)
