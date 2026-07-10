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

from evalscope.api.agent import AgentLoopResult, NativeAgentConfig
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages import ChatMessageUser
from evalscope.api.registry import get_strategy, resolve_tool_infos, resolve_tools
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

    def _native_command_timeout(self) -> Optional[float]:
        """Return the NativeAgentConfig command timeout, when explicitly configured."""
        if self._task_config is None:
            return None

        ac = self._task_config.agent_config
        if isinstance(ac, NativeAgentConfig):
            return ac.command_timeout
        return None

    def build_initial_messages(self, sample: Any) -> List[Any]:
        """Return the message list the loop starts with.

        Default: promote ``sample.input`` into a single
        :class:`ChatMessageUser` when it is a plain string, otherwise copy
        the list.
        """
        if isinstance(sample.input, list):
            return list(sample.input)
        return [ChatMessageUser(content=sample.input)]

    def build_max_steps_finalization_message(self, sample: Any) -> Optional[str]:
        """Return a no-tools finalization prompt after the loop exhausts its step budget.

        The default ``None`` keeps the standard AgentLoop result. Benchmarks whose
        official protocol requires one final model turn can override this hook.
        """
        return None

    def should_finalize_after_max_steps(self, result: AgentLoopResult) -> bool:
        """Return whether a max-steps result needs the optional final model turn."""
        return not result.final_output.completion.strip()

    def _maybe_run_external_agent(self, ac: Any, model: Any, sample: Any) -> Optional[InferenceResult]:
        if ac is None or isinstance(ac, NativeAgentConfig):
            return None

        # Local import to keep the bridge stack out of the adapter's
        # module-load-time imports (no aiohttp dependency for non-external
        # benchmark runs).
        from evalscope.agent.external.adapter import run_external_agent
        from evalscope.agent.external.config import ExternalAgentConfig

        if not isinstance(ac, ExternalAgentConfig):
            return None

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

    def _resolve_strategy(self, sample: Any, ac: Any) -> Any:
        strategy = self.build_strategy(sample)
        if not isinstance(ac, NativeAgentConfig):
            return strategy

        explicit_fields = ac.model_fields_set
        if 'strategy' not in explicit_fields and 'kwargs' not in explicit_fields:
            return strategy

        # Keep the benchmark strategy name when the user only supplies kwargs.
        strategy_name = ac.strategy if 'strategy' in explicit_fields else strategy.name
        return get_strategy(strategy_name)(**ac.kwargs)

    def _resolve_max_steps(self, ac: Any) -> int:
        if isinstance(ac, NativeAgentConfig) and 'max_steps' in ac.model_fields_set:
            return ac.max_steps
        return self.max_steps

    def _resolve_tools(self, sample: Any, ac: Any) -> tuple[Dict[str, Any], List[Any]]:
        from evalscope.agent.tools.bash import apply_bash_command_timeout_defaults

        handlers = self.build_tools(sample)
        all_tools = list(sample.tools or [])
        if not isinstance(ac, NativeAgentConfig):
            return handlers, all_tools

        if ac.tools:
            configured_handlers = resolve_tools(ac.tools)
            # Benchmark handlers win name collisions so required task semantics
            # cannot be replaced by a global config.
            handlers = {**configured_handlers, **handlers}
            existing_tool_names = {tool.name for tool in all_tools}
            for tool_info in resolve_tool_infos(ac.tools):
                if tool_info.name not in existing_tool_names:
                    all_tools.append(tool_info)
                    existing_tool_names.add(tool_info.name)

        return apply_bash_command_timeout_defaults(handlers, all_tools, ac.command_timeout)

    @staticmethod
    def _resolve_mcp_configs(ac: Any) -> Optional[List['MCPServerConfig']]:
        if isinstance(ac, NativeAgentConfig):
            return ac.mcp_servers or None
        return None

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
        from evalscope.api.agent import run_agent_loop

        ac = self._task_config.agent_config if self._task_config is not None else None
        external_result = self._maybe_run_external_agent(ac, model, sample)
        if external_result is not None:
            return external_result

        strategy = self._resolve_strategy(sample, ac)
        handlers, all_tools = self._resolve_tools(sample, ac)
        max_steps = self._resolve_max_steps(ac)
        mcp_configs = self._resolve_mcp_configs(ac)

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

        finalization_prompt = self.build_max_steps_finalization_message(sample)
        if finalization_prompt and self._reached_max_steps(result) and self.should_finalize_after_max_steps(result):
            return self._finalize_after_max_steps(model, result, finalization_prompt)

        # Resolve the final prediction through the strategy → adapter hook
        # chain so benchmarks (e.g. SWE-bench) can extract a custom payload
        # like a git patch from the trajectory.
        final_text = self._extract_final_answer(result, strategy)

        output = result.final_output
        output.completion = final_text
        return InferenceResult(output=output, messages=result.messages, trace=result.trace)

    @staticmethod
    def _reached_max_steps(result: AgentLoopResult) -> bool:
        from evalscope.api.agent import EventType
        return any(
            event.type == EventType.ERROR and event.payload.get('message') == 'max_steps_exceeded'
            for event in result.trace.events
        )

    @staticmethod
    def _finalize_after_max_steps(model: Any, result: AgentLoopResult, prompt: str) -> InferenceResult:
        from evalscope.api.agent import EventType

        finalization_message = ChatMessageUser(content=prompt)
        finalization_input = list(result.messages) + [finalization_message]
        final_output = model.generate(input=finalization_input, tools=None)
        messages = finalization_input + [final_output.message]

        step = result.trace.max_steps
        result.trace.add_event(
            step=step,
            type=EventType.NUDGE,
            message_id=finalization_message.id,
            payload={'reason': 'max_steps_finalization'},
        )
        usage = None
        if final_output.usage is not None:
            usage = {
                'input': final_output.usage.input_tokens,
                'output': final_output.usage.output_tokens,
                'total': final_output.usage.total_tokens,
            }
        result.trace.add_event(
            step=step,
            type=EventType.MODEL_GENERATE,
            message_id=final_output.message.id,
            token_usage=usage,
            payload={
                'stop_reason': final_output.stop_reason,
                'phase': 'max_steps_finalization'
            },
        )
        if final_output.completion.strip():
            result.trace.add_event(
                step=step,
                type=EventType.SUBMIT,
                message_id=final_output.message.id,
                payload={
                    'final_answer': final_output.completion,
                    'phase': 'max_steps_finalization'
                },
            )
            if result.trace.total_usage is not None and final_output.usage is not None:
                result.trace.total_usage += final_output.usage
        return InferenceResult(output=final_output, messages=messages, trace=result.trace)

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
