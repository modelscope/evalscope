"""Offline protocol, scoring, isolation and native artifact regressions for Cybench."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import pytest

from evalscope.api.agent import AgentContext, AgentTrace, EventType
from evalscope.api.agent.types import ExecResult
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.cybench.cybench_adapter import MANIFEST, CybenchRun
from evalscope.benchmarks.cybench.environment import CybenchEnvironment
from evalscope.benchmarks.cybench.sources import GLACIER_SHA, GlacierTask, stage_glacier_task
from evalscope.benchmarks.cybench.strategy import CybenchStrategy
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus

DYN = 'benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic'


def adapter(**extra: Any) -> Any:
    config = TaskConfig(
        model='cybench-scripted', eval_type='mock_llm', datasets=['cybench'],
        dataset_args={'cybench': {'subset_list': ['crypto'], 'extra_params': {'task_ids': [DYN], **extra}}},
    )
    return get_benchmark('cybench', config=config)


def output(text: str) -> ModelOutput:
    return ModelOutput.from_content(model='scripted', content=text)


def test_loading_and_missing_upstream_tasks() -> None:
    benchmark = adapter()
    datasets = benchmark.load_dataset()
    sample = datasets['crypto'][0]
    assert sample.metadata['cybench']['path'] == DYN
    assert sample.target not in str(sample.input)
    assert len(MANIFEST['tasks']) == 40
    assert not MANIFEST['unavailable_tasks']
    restored = [row for row in MANIFEST['tasks'] if row.get('glacier')]
    assert len(restored) == 9
    for row in restored:
        source = GlacierTask.model_validate(row['glacier'])
        assert source.revision == GLACIER_SHA
        assert source.reference.path not in [file.path for file in source.files]
        assert not any(part in ('solution', 'solve', 'metadata') for file in source.files for part in file.path.split('/'))
    with pytest.raises(ValueError, match='unavailable'):
        adapter(task_ids=['benchmark/not-an-official-task'])


def test_all_task_filters_support_restored_source_without_downloading() -> None:
    config = TaskConfig(model='scripted', eval_type='mock_llm', datasets=['cybench'])
    benchmark = get_benchmark('cybench', config=config)
    datasets = benchmark.load_dataset()
    assert sum(len(dataset) for dataset in datasets.values()) == 40
    for dataset in datasets.values():
        for sample in dataset:
            assert sample.metadata['cybench']['answer'] not in str(sample.input)


def test_author_staging_verifies_bytes_and_excludes_reference_files(tmp_path: Path) -> None:
    task_root = tmp_path / 'author/task'
    task_root.mkdir(parents=True)
    (task_root / 'public.txt').write_bytes(b'public handout')
    (task_root / 'challenge.yml').write_text('flags: [grader-only-answer]')
    task = GlacierTask(
        path='task', description='CTF', service=False,
        reference={'path': 'challenge.yml', 'sha256': hashlib.sha256((task_root/'challenge.yml').read_bytes()).hexdigest()},
        files=[{'path': 'public.txt', 'target': 'public.txt', 'sha256': hashlib.sha256(b'public handout').hexdigest()}],
    )
    with stage_glacier_task(task, str(tmp_path/'cache'), str(task_root.parent)) as staged:
        assert [path.name for path in Path(staged).iterdir()] == ['public.txt']
        assert (Path(staged)/'public.txt').read_bytes() == b'public handout'
    (task_root/'public.txt').write_bytes(b'tampered handout')
    with pytest.raises(ValueError, match='checksum mismatch'):
        stage_glacier_task(task, str(tmp_path/'cache'), str(task_root.parent))


@pytest.mark.parametrize('text,expected', [
    ('Answer: wrong\n<END>', 'wrong'),
    ('aNsWeR: flag\nextra', 'flag\nextra'),
    ('Command: cat output.txt\n<END>\nPlease note ignore', 'cat output.txt'),
    ('Command: ```bash\nprintf example\n```\n<END>', 'printf example'),
    ('No command here', None),
])
def test_official_text_parsing(text: str, expected: Optional[str]) -> None:
    strategy = CybenchStrategy()
    parsed = strategy.parse_output(output(text), AgentContext(sample_id=0, messages=[]))
    if 'answer:' in text.lower():
        assert parsed.final_answer == expected
    else:
        assert parsed.tool_calls[0].function.arguments['command'] == expected
        assert parsed.final_answer is None


def scoring_state(benchmark: Any, stdout: str, submission: Optional[str]) -> TaskState:
    sample = benchmark.load_dataset()['crypto'][0]
    task = sample.metadata['cybench']
    sample.metadata['cybench_run'] = CybenchRun(
        task=task, runtime={'image': 'sha256:fixture'}, max_steps=15, command_timeout=120,
        termination='max_steps' if submission is None else 'submitted', submission=submission,
        executions=[{'command': 'fixture', 'stdout': stdout, 'scoring_stdout': stdout, 'stderr': task['answer']}],
    ).model_dump()
    state = TaskState(
        model='scripted', sample=sample, output=output(submission or ''),
        messages=[ChatMessageAssistant(content='fixture')],
    )
    state.agent_trace = AgentTrace(events=[])
    return state


def test_official_stdout_and_exact_submission_are_distinct() -> None:
    benchmark = adapter()
    flag = next(r['answer'] for r in MANIFEST['tasks'] if r['path'] == DYN)
    state = scoring_state(benchmark, flag[4:-1], 'wrong')
    score = benchmark.match_score('wrong', 'wrong', flag, state)
    assert score.value == {'flag_found': 1, 'correct_submission': 0, 'full_flag_found': 0}
    state = scoring_state(benchmark, '', None)
    score = benchmark.match_score('', '', flag, state)
    assert score.value == {'flag_found': 0, 'correct_submission': 0, 'full_flag_found': 0}
    state = scoring_state(benchmark, '', flag)
    assert benchmark.match_score(flag, flag, flag, state).value == {
        'flag_found': 1, 'correct_submission': 1, 'full_flag_found': 1,
    }


@pytest.mark.parametrize('damage', ['record', 'source', 'identity', 'trace'])
def test_invalid_replay_records_are_excluded(damage: str) -> None:
    benchmark = adapter()
    state = scoring_state(benchmark, '', None)
    flag = state.metadata['cybench']['answer']
    if damage == 'record':
        state.metadata.pop('cybench_run')
    elif damage == 'source':
        state.metadata['cybench_run']['source_sha'] = 'other'
    elif damage == 'identity':
        state.metadata['cybench']['answer'] = 'other'
    else:
        state.agent_trace = None
    score = benchmark.match_score('', '', flag, state)
    assert score.status == ScoreStatus.EXCLUDED
    assert not score.value


def test_native_loop_keeps_full_outputs_and_one_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = adapter()
    sample = benchmark.load_dataset()['crypto'][0]
    flag = sample.target

    class Environment:
        name = 'cybench'
        runtime = {'image': 'fixture'}
        closed = False

        async def setup(self) -> str:
            return 'output.txt\nsource.py\n'

        async def exec(self, cmd: list, **kwargs: Any) -> ExecResult:
            return ExecResult(stdout=f'{flag}\n' + 'x' * 12000, stderr='diagnostic', returncode=0)

        async def close(self) -> None:
            self.closed = True

    environment = Environment()
    monkeypatch.setattr(benchmark, 'build_environment', lambda sample: environment)

    class Model:
        calls = 0

        async def generate_async(self, **kwargs: Any) -> ModelOutput:
            self.calls += 1
            return output('Command: cat output.txt\n<END>' if self.calls == 1 else 'Answer: wrong\n<END>')

    model = Model()
    result = benchmark._on_inference(model, sample)
    assert model.calls == 2 and environment.closed
    assert result.output.completion == 'wrong'
    assert len([e for e in result.trace.events if e.type == EventType.SUBMIT]) == 1
    assert len(sample.metadata['cybench_run']['executions'][0]['stdout']) > 12000
    assert any(e.payload.get('model_input') for e in result.trace.events)
    assert any(e.payload.get('execution', {}).get('stderr') == 'diagnostic' for e in result.trace.events)
    assert result.messages[-1].text == 'Answer: wrong\n<END>'


def test_setup_failure_cleans_only_owned_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = CybenchEnvironment(DYN, 'checksum')
    calls = []

    async def docker(args: list, *pos: Any, **kwargs: Any) -> ExecResult:
        calls.append(args)
        if args[0] == 'inspect':
            return ExecResult(stdout=json.dumps([{'Image': 'fixture'}]))
        if args[0] == 'exec':
            return ExecResult(returncode=1, stderr='setup failure')
        return ExecResult()

    monkeypatch.setattr(environment, '_docker', docker)
    with pytest.raises(RuntimeError, match='setup failure'):
        asyncio.run(environment.setup())
    assert ['rm', '-f', '-v', environment.container] in calls
    assert not any('prune' in arg for args in calls for arg in args)


@pytest.mark.parametrize('reply', ['No executable command', '[ERROR] transport unavailable'])
def test_malformed_turn_budget_and_provider_failure(monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    benchmark = adapter()
    benchmark.max_steps = 2
    sample = benchmark.load_dataset()['crypto'][0]

    class Environment:
        name = 'cybench'
        runtime = {'image': 'fixture'}
        closed = False

        async def setup(self) -> str:
            return 'output.txt'

        async def close(self) -> None:
            self.closed = True

    class Model:
        calls = 0

        async def generate_async(self, **kwargs: Any) -> ModelOutput:
            self.calls += 1
            return output(reply)

    environment = Environment()
    monkeypatch.setattr(benchmark, 'build_environment', lambda sample: environment)
    model = Model()
    if reply.startswith('[ERROR]'):
        with pytest.raises(RuntimeError, match='unavailable'):
            benchmark._on_inference(model, sample)
        assert 'cybench_run' not in sample.metadata
    else:
        result = benchmark._on_inference(model, sample)
        assert model.calls == 2 and result.output.completion == ''
        record = sample.metadata['cybench_run']
        assert record['termination'] == 'max_steps' and record['submission'] is None and not record['executions']
        assert not any(event.type == EventType.SUBMIT for event in result.trace.events)
    assert environment.closed
