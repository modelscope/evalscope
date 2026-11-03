"""Offline protocol, scoring, isolation and native artifact regressions for Cybench."""

import asyncio
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Optional

import pytest

from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
from evalscope.api.agent import AgentContext, AgentTrace, EventType
from evalscope.api.agent.types import ExecResult
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.cybench.cybench_adapter import SOURCE_SHA, CybenchRun
from evalscope.benchmarks.cybench.dataset import stage_task_archive
from evalscope.benchmarks.cybench.environment import CybenchEnvironment
from evalscope.benchmarks.cybench.strategy import CybenchStrategy
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus

DYN = 'benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic'
FLAG = 'HTB{scripted_flag}'


def write_dataset(root: Path) -> Path:
    """Build a tiny local dataset instead of depending on network or redistributing the official catalog."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (path, category, statement) in enumerate([
        (DYN, 'crypto', ''), ('benchmark/restored/crypto/example', 'crypto', 'Original competition statement'),
        ('benchmark/restored/web/example', 'web', 'Original service statement'),
    ]):
        rows.append({
            'path': path, 'category': category, 'difficulty': 'unrated', 'target_host': '',
            'answer': FLAG, 'answer_format': 'HTB{...}', 'metadata_sha256': '0' * 64,
            'archive': f'tasks/{index:02d}.tar.gz', 'archive_sha256': '1' * 64, 'statement': statement,
        })
    (root / 'tasks.json').write_text(json.dumps({'source_sha': SOURCE_SHA, 'tasks': rows}))
    return root


@pytest.fixture(autouse=True)
def local_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.api.dataset import download_dataset_file
    from evalscope.benchmarks.cybench import cybench_adapter

    root = write_dataset(tmp_path / 'dataset')

    def download(dataset_id: str, path: str, **kwargs: Any) -> str:
        assert dataset_id == 'evalscope/cybench'
        assert kwargs['data_source'] == 'modelscope'
        return download_dataset_file(str(root), path)

    monkeypatch.setattr(cybench_adapter, 'download_dataset_file', download)


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
    assert len(benchmark.task_manifest.tasks) == 3
    assert benchmark.task_manifest.tasks[1].statement == 'Original competition statement'
    with pytest.raises(ValueError, match='unavailable'):
        adapter(task_ids=['benchmark/not-an-official-task']).load_dataset()


def test_all_task_filters_support_restored_source_without_downloading() -> None:
    config = TaskConfig(model='scripted', eval_type='mock_llm', datasets=['cybench'])
    benchmark = get_benchmark('cybench', config=config)
    datasets = benchmark.load_dataset()
    assert sum(len(dataset) for dataset in datasets.values()) == 3
    for dataset in datasets.values():
        for sample in dataset:
            assert sample.metadata['cybench']['answer'] not in str(sample.input)


def test_hub_downloads_catalog_once_and_only_the_selected_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip('ms_enclave')
    from evalscope.benchmarks.cybench import cybench_adapter

    download = cybench_adapter.download_dataset_file
    calls = []

    def tracked_download(dataset_id: str, path: str, **kwargs: Any) -> str:
        calls.append(path)
        if path == 'tasks.json':
            return download(dataset_id, path, **kwargs)
        return str(tmp_path / 'selected.tar.gz')

    monkeypatch.setattr(cybench_adapter, 'download_dataset_file', tracked_download)
    benchmark = adapter()
    sample = benchmark.load_dataset()['crypto'][0]
    environment = benchmark.build_environment(sample)
    assert calls == ['tasks.json', sample.metadata['cybench']['archive']]
    assert environment.archive_path == str(tmp_path / 'selected.tar.gz')


def test_shared_sandbox_defaults_override_runtime_image(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('ms_enclave')
    config = TaskConfig(
        model='scripted', eval_type='mock_llm', datasets=['cybench'],
        sandbox={'default_config': {'image': 'selected:runtime', 'memory_limit': '8g', 'cpu_limit': 4}},
    )
    benchmark = get_benchmark('cybench', config=config)
    sample = benchmark.load_dataset()['crypto'][0]
    monkeypatch.setattr(benchmark, '_dataset_file', lambda path: '/selected/task.tar.gz')
    environment = benchmark.build_environment(sample)
    assert environment.image == 'selected:runtime'
    assert environment._sandbox_config_dict['memory_limit'] == '8g'
    assert environment._sandbox_config_dict['cpu_limit'] == 4


def test_archive_staging_verifies_bytes_and_rejects_unsafe_members(tmp_path: Path) -> None:
    path = tmp_path / 'task.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        item = tarfile.TarInfo('public.txt')
        item.size = len(b'public handout')
        archive.addfile(item, io.BytesIO(b'public handout'))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with stage_task_archive(str(path), checksum) as staged:
        assert (Path(staged) / 'public.txt').read_bytes() == b'public handout'
    with pytest.raises(ValueError, match='checksum mismatch'):
        stage_task_archive(str(path), 'wrong')
    for name in ('../escape', '/absolute', 'symlink'):
        with tarfile.open(path, 'w:gz') as archive:
            item = tarfile.TarInfo(name)
            if name == 'symlink':
                item.type = tarfile.SYMTYPE
                item.linkname = '/grader'
            archive.addfile(item)
        with pytest.raises(ValueError, match='Unsafe'):
            stage_task_archive(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


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
    flag = FLAG
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


@pytest.mark.parametrize('timed_out', [False, True])
def test_native_loop_keeps_full_outputs_and_one_submission(
    monkeypatch: pytest.MonkeyPatch, timed_out: bool
) -> None:
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
            return ExecResult(stdout=f'{flag}\n' + 'x' * 12000, stderr='diagnostic', timed_out=timed_out)

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
    scoring_stdout = sample.metadata['cybench_run']['executions'][0]['scoring_stdout']
    assert (flag in scoring_stdout) is not timed_out
    assert any(e.payload.get('model_input') for e in result.trace.events)
    assert any(e.payload.get('execution', {}).get('stderr') == 'diagnostic' for e in result.trace.events)
    assert result.messages[-1].text == 'Answer: wrong\n<END>'


def test_setup_failure_cleans_only_owned_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip('ms_enclave')
    task = adapter().task_manifest.tasks[0]
    path = tmp_path / 'task.tar.gz'
    with tarfile.open(path, 'w:gz'):
        pass
    task.archive_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    environment = CybenchEnvironment(task, str(path))
    closed = []

    async def identity() -> dict:
        return {'image': 'fixture', 'manager': 'ms-enclave'}

    async def fail(args: list, *pos: Any, **kwargs: Any) -> ExecResult:
        return ExecResult(returncode=1, stderr='setup failure')

    async def close() -> None:
        closed.append(True)

    monkeypatch.setattr(environment, '_runtime_identity', identity)
    monkeypatch.setattr(environment, '_root_exec', fail)
    monkeypatch.setattr(environment, 'close', close)
    with pytest.raises(RuntimeError, match='setup failure'):
        asyncio.run(environment.setup())
    assert closed == [True]


@pytest.mark.parametrize('timed_out', [False, True])
def test_ms_enclave_owns_runtime_and_public_commands(monkeypatch: pytest.MonkeyPatch, timed_out: bool) -> None:
    pytest.importorskip('ms_enclave')
    environment = CybenchEnvironment(adapter().task_manifest.tasks[0], '/selected/task.tar.gz')
    assert isinstance(environment, EnclaveAgentEnvironment)
    config = environment._sandbox_config_dict
    assert config['image'] == environment.image and config['privileged']
    assert config['tools_config'] == ['shell_executor']
    assert not config.get('volumes')
    calls = []

    async def execute(self: Any, cmd: list, **kwargs: Any) -> ExecResult:
        calls.append((cmd, kwargs))
        return ExecResult(stdout='public result', stderr='diagnostic', returncode=124, timed_out=timed_out)

    monkeypatch.setattr(EnclaveAgentEnvironment, 'exec', execute)
    result = asyncio.run(environment.exec(['bash', '-c', 'sleep 30'], input='payload', timeout=1, env={'X': 'value'}))
    command, kwargs = calls[0]
    assert command[:4] == ['runuser', '-u', 'cybench-agent', '--']
    assert kwargs == {'cwd': '/tmp/cyber-bench', 'input': 'payload', 'timeout': 1, 'env': {'X': 'value'}}
    assert result.timed_out is timed_out and result.stdout == 'public result' and result.stderr == 'diagnostic'


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
