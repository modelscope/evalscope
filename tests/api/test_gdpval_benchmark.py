import asyncio
import base64
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.agent.types import ExecResult
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.benchmarks.gdpval.gdpval_adapter import (
    GDPvalOpenAIGoldAdapter,
    _GDPvalArtifactEnvironment,
    _relative_deliverable_path,
)
from evalscope.config import TaskConfig
from evalscope.constants import HubType


def make_adapter(**extra_params: Any) -> GDPvalOpenAIGoldAdapter:
    base_extra_params = {
        'dataset_hub': HubType.MODELSCOPE,
        'dataset_revision': '',
        'max_steps': 250,
        'command_timeout': 180.0,
        'docker_image': 'evalscope/gdpval:latest',
        'auto_build_docker_image': True,
        'network_enabled': True,
        'download_reference_files': False,
        'scoring_mode': 'export_only',
    }
    base_extra_params.update(extra_params)
    meta = BenchmarkMeta(
        name='gdpval_openai_gold',
        dataset_id='openai-mirror/gdpval',
        subset_list=['default'],
        default_subset='default',
        eval_split='train',
        prompt_template='{question}',
        metric_list=['submission_ready'],
        extra_params=base_extra_params,
    )
    cfg = TaskConfig(
        datasets=['gdpval_openai_gold'],
        dataset_args={'gdpval_openai_gold': {
            'extra_params': extra_params
        }},
    )
    return GDPvalOpenAIGoldAdapter(benchmark_meta=meta, task_config=cfg)


def test_record_to_sample_uses_modelscope_metadata_and_prompt() -> None:
    adapter = make_adapter()
    sample = adapter.record_to_sample({
        'task_id': 'task-1',
        'sector': 'Finance',
        'occupation': 'Analyst',
        'prompt': 'Create the workbook.',
        'reference_files': ['reference_files/abc123/input.xlsx'],
        'reference_file_urls': ['https://example.test/input.xlsx'],
        'reference_file_hf_uris': ['hf://datasets/openai/gdpval/reference_files/abc123/input.xlsx'],
        'rubric_pretty': 'Rubric',
        'rubric_json': {
            'criteria': []
        },
    })

    assert 'Create the workbook.' in sample.input
    assert '/reference_files/abc123/input.xlsx' in sample.input
    assert 'deliverable_files' in sample.input
    assert sample.metadata['dataset_id'] == 'openai-mirror/gdpval'
    assert sample.metadata['dataset_hub'] == HubType.MODELSCOPE
    assert sample.metadata['reference_paths'] == ['reference_files/input.xlsx']
    assert sample.metadata['sandbox_reference_paths'] == ['/reference_files/abc123/input.xlsx']


def test_build_reference_volumes_uses_downloaded_file_parents(tmp_path: Path) -> None:
    adapter = make_adapter()
    host_dir = tmp_path / 'reference_files' / 'abc123'
    host_dir.mkdir(parents=True)
    host_file = host_dir / 'input.xlsx'
    host_file.write_bytes(b'data')
    sample = Sample(input='prompt', metadata={'host_reference_files': [str(host_file)]})

    volumes = adapter._build_reference_volumes(sample)

    assert volumes[str(host_dir)] == {'bind': '/reference_files/abc123', 'mode': 'ro'}


def test_relative_deliverable_path_rejects_unsafe_paths() -> None:
    assert _relative_deliverable_path('deliverable_files/report.pdf') == 'report.pdf'
    assert _relative_deliverable_path('deliverable_files/nested/report.pdf') == 'nested/report.pdf'
    assert _relative_deliverable_path('/tmp/report.pdf') == ''
    assert _relative_deliverable_path('deliverable_files/../report.pdf') == ''


def test_artifact_environment_extracts_deliverables(tmp_path: Path) -> None:
    metadata: Dict[str, Any] = {}
    fake_env = FakeEnvironment({
        'deliverable_files/report.txt': b'hello',
        'deliverable_files/nested/table.csv': b'a,b\n1,2\n',
    })
    env = _GDPvalArtifactEnvironment(env=fake_env, artifact_dir=tmp_path, metadata=metadata)

    asyncio.run(env.close())

    assert (tmp_path / 'deliverable_files/report.txt').read_bytes() == b'hello'
    assert (tmp_path / 'deliverable_files/nested/table.csv').read_bytes() == b'a,b\n1,2\n'
    assert metadata['deliverable_files'] == [
        {
            'path': 'deliverable_files/report.txt',
            'local_path': str(tmp_path / 'deliverable_files/report.txt'),
        },
        {
            'path': 'deliverable_files/nested/table.csv',
            'local_path': str(tmp_path / 'deliverable_files/nested/table.csv'),
        },
    ]


def test_match_score_marks_submission_ready_with_deliverable() -> None:
    adapter = make_adapter()
    sample = Sample(
        input='prompt',
        target='',
        metadata={'deliverable_files': [{
            'path': 'deliverable_files/report.txt'
        }]},
    )
    state = TaskState(model='mock', sample=sample, completed=True)

    score = adapter.match_score('', '', '', state)

    assert score.value['submission_ready'] == 1.0
    assert score.metadata['deliverable_count'] == 1


def test_ensure_docker_image_builds_missing_default_image(monkeypatch: Any) -> None:
    adapter = make_adapter()
    calls: List[List[str]] = []

    def fake_run(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(cmd)
        if cmd[:3] == ['docker', 'image', 'inspect']:
            return subprocess.CompletedProcess(cmd, 1)
        if cmd[:2] == ['docker', 'build']:
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f'unexpected command: {cmd}')

    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/docker' if name == 'docker' else None)
    monkeypatch.setattr('subprocess.run', fake_run)

    adapter._ensure_docker_image()
    adapter._ensure_docker_image()

    assert calls[0] == ['docker', 'image', 'inspect', 'evalscope/gdpval:latest']
    assert calls[1][:4] == ['docker', 'build', '-t', 'evalscope/gdpval:latest']
    assert len(calls) == 2


def test_ensure_docker_image_skips_custom_image(monkeypatch: Any) -> None:
    adapter = make_adapter(docker_image='custom/gdpval:latest')

    def fail_run(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        raise AssertionError(f'unexpected command: {cmd}')

    monkeypatch.setattr('subprocess.run', fail_run)

    adapter._ensure_docker_image()


class FakeEnvironment:
    name = 'fake'

    def __init__(self, files: Dict[str, bytes]) -> None:
        self.files = files
        self.closed = False

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        if cmd[:3] == ['test', '-d', 'deliverable_files']:
            return ExecResult(returncode=0, stdout='', stderr='')
        if cmd[:4] == ['find', 'deliverable_files', '-type', 'f']:
            return ExecResult(returncode=0, stdout='\0'.join(self.files.keys()) + '\0', stderr='')
        if cmd[:3] == ['base64', '-w', '0']:
            return ExecResult(returncode=0, stdout=base64.b64encode(self.files[cmd[3]]).decode(), stderr='')
        return ExecResult(returncode=1, stdout='', stderr='unexpected command')

    async def close(self) -> None:
        self.closed = True
