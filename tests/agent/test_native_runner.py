from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.agent.runner import run_native_agent
from evalscope.agent.tools.bash import BASH_TOOL_INFO
from evalscope.api.agent import AgentLoopResult, AgentTrace
from evalscope.api.agent.runner import run_agent_loop
from evalscope.api.agent.types import ExecResult, NativeAgentConfig
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.config import TaskConfig


class FakeEnvironment:

    name = 'fake'

    def __init__(self, *, exec_returncode: int = 0, timeout: Optional[float] = None, **_: Any) -> None:
        self.exec_returncode = exec_returncode
        self.timeout = timeout
        self.closed = 0
        self.put_dirs: List[tuple[str, str]] = []

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        return ExecResult(returncode=self.exec_returncode, stderr='install failed')

    async def put_dir(self, source_dir: str | Path, target_dir: str) -> None:
        self.put_dirs.append((str(source_dir), target_dir))

    async def close(self) -> None:
        self.closed += 1


class FakeStrategy:

    def __init__(self, **_: Any) -> None:
        pass


def test_run_native_agent_keeps_environment_override_open(monkeypatch: pytest.MonkeyPatch) -> None:
    env = FakeEnvironment()
    seen: Dict[str, Any] = {}

    def fake_run_agent_loop(**kwargs: Any) -> AgentLoopResult:
        seen['environment'] = kwargs['environment']
        seen['close_environment'] = kwargs['close_environment']
        if kwargs['close_environment']:
            asyncio.run(kwargs['environment'].close())
        return AgentLoopResult(
            messages=[ChatMessageAssistant(content='raw')],
            final_output=_model_output('raw'),
            trace=AgentTrace(strategy='fake', environment='fake', max_steps=1),
        )

    monkeypatch.setattr('evalscope.agent.runner.get_strategy', lambda name: FakeStrategy)
    monkeypatch.setattr(
        'evalscope.agent.runner.get_environment',
        lambda name: (_ for _ in ()).throw(AssertionError('override should skip environment lookup')),
    )
    monkeypatch.setattr('evalscope.agent.runner.resolve_tools', lambda tools: {})
    monkeypatch.setattr('evalscope.agent.runner.resolve_tool_infos', lambda tools: [])
    monkeypatch.setattr('evalscope.agent.runner.run_agent_loop', fake_run_agent_loop)

    result = run_native_agent(
        task_config=TaskConfig(
            datasets=['demo'],
            agent_config=NativeAgentConfig(strategy='fake', environment='missing-env', max_steps=1),
        ),
        model=object(),
        sample=Sample(id=1, input='do work', target='', metadata={}),
        build_sandbox_config=lambda _: None,
        extract_final_answer=lambda loop_result, strategy: 'final',
        environment_override=env,
    )

    assert seen['environment'] is env
    assert seen['close_environment'] is False
    assert env.closed == 0
    assert result.output.message.text == 'final'


def test_run_native_agent_applies_command_timeout_to_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Dict[str, Any] = {}
    seen_args: List[Dict[str, Any]] = []

    async def fake_bash(call: ToolCall, env: Any) -> str:
        seen_args.append(call.function.arguments)
        return 'ok'

    def fake_run_agent_loop(**kwargs: Any) -> AgentLoopResult:
        seen.update(kwargs)
        return AgentLoopResult(
            messages=[ChatMessageAssistant(content='raw')],
            final_output=_model_output('raw'),
            trace=AgentTrace(strategy='fake', max_steps=1),
        )

    monkeypatch.setattr('evalscope.agent.runner.get_strategy', lambda name: FakeStrategy)
    monkeypatch.setattr('evalscope.agent.runner.resolve_tools', lambda tools: {'bash': fake_bash})
    monkeypatch.setattr('evalscope.agent.runner.resolve_tool_infos', lambda tools: [BASH_TOOL_INFO])
    monkeypatch.setattr('evalscope.agent.runner.run_agent_loop', fake_run_agent_loop)

    run_native_agent(
        task_config=TaskConfig(
            datasets=['demo'],
            agent_config=NativeAgentConfig(
                strategy='fake',
                tools=['bash'],
                max_steps=1,
                command_timeout=120,
            ),
        ),
        model=object(),
        sample=Sample(id=1, input='do work', target='', metadata={}),
        build_sandbox_config=lambda _: None,
        extract_final_answer=lambda loop_result, strategy: 'final',
    )

    bash_schema = next(tool for tool in seen['all_tools'] if tool.name == 'bash')
    assert bash_schema.parameters.properties['timeout'].default == 120
    assert BASH_TOOL_INFO.parameters.properties['timeout'].default == 60

    wrapped_bash = seen['handlers']['bash']
    asyncio.run(wrapped_bash(ToolCall(id='1', function=ToolFunction(name='bash', arguments={'command': 'pwd'})), None))
    asyncio.run(
        wrapped_bash(
            ToolCall(id='2', function=ToolFunction(name='bash', arguments={
                'command': 'pwd',
                'timeout': 5,
            })),
            None,
        )
    )

    assert seen_args[0]['timeout'] == 120
    assert seen_args[1]['timeout'] == 5


def test_run_native_agent_passes_command_timeout_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Dict[str, Any] = {}

    def fake_run_agent_loop(**kwargs: Any) -> AgentLoopResult:
        seen.update(kwargs)
        return AgentLoopResult(
            messages=[ChatMessageAssistant(content='raw')],
            final_output=_model_output('raw'),
            trace=AgentTrace(strategy='fake', environment='fake', max_steps=1),
        )

    monkeypatch.setattr('evalscope.agent.runner.get_strategy', lambda name: FakeStrategy)
    monkeypatch.setattr('evalscope.agent.runner.get_environment', lambda name: FakeEnvironment)
    monkeypatch.setattr('evalscope.agent.runner.resolve_tools', lambda tools: {})
    monkeypatch.setattr('evalscope.agent.runner.resolve_tool_infos', lambda tools: [])
    monkeypatch.setattr('evalscope.agent.runner.run_agent_loop', fake_run_agent_loop)

    run_native_agent(
        task_config=TaskConfig(
            datasets=['demo'],
            agent_config=NativeAgentConfig(
                strategy='fake',
                environment='fake-env',
                max_steps=1,
                command_timeout=42,
            ),
        ),
        model=object(),
        sample=Sample(id=1, input='do work', target='', metadata={}),
        build_sandbox_config=lambda _: None,
        extract_final_answer=lambda loop_result, strategy: 'final',
    )

    assert seen['environment'].timeout == 42


def test_run_native_agent_closes_owned_environment_when_skill_install_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    skill = tmp_path / 'skills' / 'demo'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text(
        """---
name: demo
description: Demo skill.
---
""",
        encoding='utf-8',
    )
    env = FakeEnvironment(exec_returncode=1)

    monkeypatch.setattr('evalscope.agent.runner.get_strategy', lambda name: FakeStrategy)
    monkeypatch.setattr('evalscope.agent.runner.get_environment', lambda name: lambda **kwargs: env)
    monkeypatch.setattr('evalscope.agent.runner.resolve_tools', lambda tools: {})
    monkeypatch.setattr('evalscope.agent.runner.resolve_tool_infos', lambda tools: [])
    monkeypatch.setattr(
        'evalscope.agent.runner.run_agent_loop',
        lambda **kwargs: (_ for _ in ()).throw(AssertionError('run_agent_loop should not start')),
    )

    with pytest.raises(RuntimeError, match='NativeAgentRunner failed to install skills'):
        run_native_agent(
            task_config=TaskConfig(
                datasets=['demo'],
                agent_config=NativeAgentConfig(
                    strategy='fake',
                    environment='fake-env',
                    skills_dir=str(tmp_path / 'skills'),
                ),
            ),
            model=object(),
            sample=Sample(id=1, input='do work', target='', metadata={}),
            build_sandbox_config=lambda _: None,
            extract_final_answer=lambda loop_result, strategy: 'final',
        )

    assert env.closed == 1


def test_run_agent_loop_can_leave_caller_owned_environment_open(monkeypatch: pytest.MonkeyPatch) -> None:
    env = FakeEnvironment()

    class FakeLoop:

        def __init__(self, **_: Any) -> None:
            pass

        async def run(self, ctx: Any) -> AgentLoopResult:
            return AgentLoopResult(
                messages=[ChatMessageAssistant(content='raw')],
                final_output=_model_output('raw'),
                trace=AgentTrace(strategy='fake', environment='fake', max_steps=1),
            )

    monkeypatch.setattr('evalscope.api.agent.runner.AgentLoop', FakeLoop)

    result = run_agent_loop(
        model=object(),
        strategy=FakeStrategy(),
        handlers={},
        environment=env,
        initial_messages=[],
        all_tools=[],
        max_steps=1,
        sample_id=1,
        trace_strategy_name='fake',
        trace_env_name='fake',
        close_environment=False,
    )

    assert result.final_output.message.text == 'raw'
    assert env.closed == 0


def _model_output(text: str) -> ModelOutput:
    return ModelOutput(model='fake', choices=[ChatCompletionChoice.from_content(text)])
