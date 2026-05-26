"""Native agent loop runner.

Drives a sample through the native AgentLoop stack — strategy / tools /
environment resolution + :func:`run_agent_loop` invocation — so that
:meth:`DefaultDataAdapter._on_agent_inference` reduces to a one-line
delegate that mirrors :func:`run_external_agent`.

This module is orchestration-only.  Adapter-specific hooks
(``build_sandbox_config``, final-answer extraction) are passed in as
callables so the adapter remains the single source of truth for
per-benchmark overrides.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from evalscope.api.agent import AgentEnvironment, AgentLoopResult
from evalscope.api.benchmark.adapters._agent_loop_runner import run_agent_loop
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import Model
from evalscope.api.registry import get_environment, get_strategy, resolve_tool_infos, resolve_tools
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.dataset import Sample
    from evalscope.config import TaskConfig

logger = get_logger()


def run_native_agent(
    *,
    task_config: 'TaskConfig',
    model: Model,
    sample: 'Sample',
    build_sandbox_config: Callable[['Sample'], Optional[Dict[str, Any]]],
    extract_final_answer: Callable[[AgentLoopResult, Any], str],
) -> InferenceResult:
    """Drive a sample through the native AgentLoop and return its result.

    Mirrors the shape of :func:`run_external_agent`: returns an
    :class:`InferenceResult` bundling the final ``ModelOutput`` with the
    loop's full message trail and structured :class:`AgentTrace`.

    Args:
        task_config: Carries both ``agent_config`` (strategy / tools / env)
            and ``sandbox`` (engine / default_config / manager_config).
        model: Driving :class:`Model`.
        sample: Sample being processed.
        build_sandbox_config: Adapter hook returning a per-sample sandbox
            config dict; merged on top of
            ``task_config.sandbox.default_config``.  May return ``None``.
        extract_final_answer: Adapter hook returning the final prediction
            string from the completed loop result.  Receives the loop
            result and the strategy instance.
    """
    cfg = task_config.agent_config
    strategy_cls = get_strategy(cfg.strategy)
    strategy = strategy_cls(**cfg.kwargs)

    handlers = resolve_tools(cfg.tools)

    # Resolve ToolInfo schemas from the registry so the model can see them.
    registered_tool_infos = resolve_tool_infos(cfg.tools)

    # Determine environment class (if any) – instantiated below so its
    # constructor sees the fully merged kwargs.
    env_cls: Optional[type] = None
    if cfg.environment is not None:
        env_cls = get_environment(cfg.environment)

    env_kwargs = _resolve_env_kwargs(
        task_config=task_config,
        sample=sample,
        build_sandbox_config=build_sandbox_config,
    )
    environment: Optional[AgentEnvironment] = env_cls(**env_kwargs) if env_cls is not None else None

    if isinstance(sample.input, list):
        initial_messages = list(sample.input)
    else:
        initial_messages = [ChatMessageUser(content=sample.input)]

    # Merge sample-level tools with agent-config tools.
    sample_tools = list(sample.tools or [])
    all_tools = sample_tools + [t for t in registered_tool_infos if t not in sample_tools]

    result: AgentLoopResult = run_agent_loop(
        model=model,
        strategy=strategy,
        handlers=handlers,
        environment=environment,
        initial_messages=initial_messages,
        all_tools=all_tools,
        max_steps=cfg.max_steps,
        sample_id=sample.id,
        trace_strategy_name=cfg.strategy,
        trace_env_name=cfg.environment,
        mcp_configs=list(cfg.mcp_servers) or None,
    )

    final_text = extract_final_answer(result, strategy)
    output = result.final_output
    output.completion = final_text  # normalise content via existing setter
    return InferenceResult(output=output, messages=result.messages, trace=result.trace)


def _resolve_env_kwargs(
    *,
    task_config: 'TaskConfig',
    sample: 'Sample',
    build_sandbox_config: Callable[['Sample'], Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Merge task-level + per-sample sandbox config into env constructor kwargs.

    Precedence (lowest -> highest):
      1. ``task_config.sandbox`` — engine / default_config / manager_config
         carried alongside the pooled SandboxMixin so sandbox settings are
         defined **once** at the task level.
      2. ``build_sandbox_config(sample)`` — per-sample override hook.
      3. ``agent_config.environment_extra`` — raw kwargs forwarded verbatim
         to the environment constructor (last word for power users).
    """
    env_kwargs: Dict[str, Any] = {}
    base_sandbox_cfg: Dict[str, Any] = {}

    task_sandbox = task_config.sandbox
    if task_sandbox is not None and task_sandbox.enabled:
        env_kwargs['engine'] = task_sandbox.engine
        base_sandbox_cfg = dict(task_sandbox.default_config or {})
        if task_sandbox.manager_config:
            env_kwargs['manager_config'] = dict(task_sandbox.manager_config)

    try:
        per_sample_cfg = build_sandbox_config(sample) or {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f'build_sandbox_config raised {exc!r}; falling back to task-level sandbox config.')
        per_sample_cfg = {}

    merged_sandbox_cfg: Dict[str, Any] = {**base_sandbox_cfg, **per_sample_cfg}
    if merged_sandbox_cfg:
        env_kwargs['sandbox_config'] = merged_sandbox_cfg

    # environment_extra wins over everything above.
    env_kwargs.update(task_config.agent_config.environment_extra)
    return env_kwargs
