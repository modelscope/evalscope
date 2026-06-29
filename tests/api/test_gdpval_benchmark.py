import asyncio
import base64
import json
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.agent.types import ExecResult
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.gdpval.gdpval_adapter import (
    GDPvalAdapter,
    _GDPvalArtifactEnvironment,
    _relative_deliverable_path,
)
from evalscope.config import TaskConfig
from evalscope.constants import HubType


def make_adapter(**extra_params: Any) -> GDPvalAdapter:
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
        name='gdpval',
        dataset_id='openai-mirror/gdpval',
        subset_list=['default'],
        default_subset='default',
        eval_split='train',
        prompt_template='{question}',
        metric_list=['submission_ready', 'llm_rubric_score'],
        extra_params=base_extra_params,
    )
    cfg = TaskConfig(
        datasets=['gdpval'],
        dataset_args={'gdpval': {
            'extra_params': extra_params
        }},
    )
    return GDPvalAdapter(benchmark_meta=meta, task_config=cfg)


def test_gdpval_registered_under_short_name() -> None:
    cfg = TaskConfig(datasets=['gdpval'], dataset_args={'gdpval': {'extra_params': {'download_reference_files': False}}})

    adapter = get_benchmark('gdpval', cfg)

    assert isinstance(adapter, GDPvalAdapter)
    assert adapter.name == 'gdpval'


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


def test_resolve_reference_files_skips_empty_download(monkeypatch: Any) -> None:
    adapter = make_adapter()
    sample = Sample(input='prompt', metadata={'reference_files': ['missing.xlsx']})

    class FakeDataset:

        @staticmethod
        def download_file(file_path: str) -> Optional[str]:
            return None

    monkeypatch.setattr(GDPvalAdapter, 'source_dataset', property(lambda self: FakeDataset()))

    adapter._resolve_sample_reference_files([sample])

    assert sample.metadata['host_reference_files'] == []


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


def test_artifact_environment_handles_listing_failure(tmp_path: Path) -> None:
    metadata: Dict[str, Any] = {}

    class ListingFailureEnvironment(FakeEnvironment):

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
                return ExecResult(returncode=1, stdout='', stderr='find failed')
            return await super().exec(cmd, cwd=cwd, input=input, timeout=timeout, env=env)

    env = _GDPvalArtifactEnvironment(env=ListingFailureEnvironment({}), artifact_dir=tmp_path, metadata=metadata)

    asyncio.run(env.close())

    assert metadata['deliverable_files'] == []
    assert metadata['artifact_dir'] == str(tmp_path)


def test_artifact_environment_skips_failed_base64_extract(tmp_path: Path) -> None:
    metadata: Dict[str, Any] = {}

    class Base64FailureEnvironment(FakeEnvironment):

        async def exec(
            self,
            cmd: List[str],
            *,
            cwd: Optional[str] = None,
            input: Optional[str] = None,
            timeout: Optional[float] = None,
            env: Optional[Dict[str, str]] = None,
        ) -> ExecResult:
            if cmd[:3] == ['base64', '-w', '0']:
                return ExecResult(returncode=1, stdout='', stderr='base64 failed')
            return await super().exec(cmd, cwd=cwd, input=input, timeout=timeout, env=env)

    env = _GDPvalArtifactEnvironment(
        env=Base64FailureEnvironment({'deliverable_files/report.txt': b'hello'}),
        artifact_dir=tmp_path,
        metadata=metadata,
    )

    asyncio.run(env.close())

    assert metadata['deliverable_files'] == []
    assert metadata['artifact_dir'] == str(tmp_path)


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


def test_match_score_runs_non_official_llm_rubric_judge() -> None:
    adapter = make_adapter(scoring_mode='llm_rubric_judge')
    fake_judge = FakeLLMJudge()
    adapter.llm_judge = fake_judge
    sample = Sample(
        input='Task prompt',
        target='',
        metadata={
            'rubric_pretty': 'Rubric text',
            'rubric_json': '[{"criterion": "include a report", "score": 1}]',
            'deliverable_files': [{
                'path': 'deliverable_files/report.txt'
            }],
        },
    )
    state = TaskState(model='mock', sample=sample, completed=True)

    score = adapter.match_score('Done.', 'Done.', '', state)

    assert score.main_score_name == 'llm_rubric_score'
    assert score.value['submission_ready'] == 1.0
    assert score.value['llm_rubric_score'] == 0.75
    assert score.metadata['llm_rubric_score']['non_official_score'] is True
    assert score.metadata['llm_rubric_score']['official_gdpval_score_computed_locally'] is False
    assert 'Rubric text' in fake_judge.prompt
    assert 'deliverable_files/report.txt' in fake_judge.prompt


def test_ensure_docker_image_builds_missing_default_image(monkeypatch: Any) -> None:
    adapter = make_adapter()
    calls: List[Dict[str, Any]] = []

    def mock_ensure(image: str, path: str, dockerfile: str, label: str) -> bool:
        calls.append({'image': image, 'path': path, 'dockerfile': dockerfile, 'label': label})
        return True

    monkeypatch.setattr('evalscope.benchmarks.gdpval.gdpval_adapter.ensure_docker_image_built', mock_ensure)

    adapter._ensure_docker_image()
    adapter._ensure_docker_image()

    assert calls == [{
        'image': 'evalscope/gdpval:latest',
        'path': str(Path('evalscope/benchmarks/gdpval').resolve()),
        'dockerfile': str(Path('evalscope/benchmarks/gdpval/Dockerfile').resolve()),
        'label': 'GDPval docker image',
    }]


def test_ensure_docker_image_skips_custom_image(monkeypatch: Any) -> None:
    adapter = make_adapter(docker_image='custom/gdpval:latest')

    monkeypatch.setattr(
        'evalscope.benchmarks.gdpval.gdpval_adapter.ensure_docker_image_built',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(f'unexpected image build: {args} {kwargs}')),
    )

    adapter._ensure_docker_image()


def test_export_submission_writes_parquet_and_copies_deliverables(tmp_path: Path) -> None:
    adapter = make_adapter()
    adapter._submission_records = [{
        'task_id': 'task-1',
        'prompt': 'Create a report.',
        'sector': 'Finance',
        'occupation': 'Analyst',
        'reference_files': [],
        'reference_file_urls': [],
        'reference_file_hf_uris': [],
    }]
    source_file = tmp_path / 'source' / 'report.txt'
    source_file.parent.mkdir()
    source_file.write_text('hello', encoding='utf-8')

    report_dir = tmp_path / 'reports' / 'qwen-plus'
    review_dir = tmp_path / 'reviews' / 'qwen-plus'
    review_dir.mkdir(parents=True)
    review_item = {
        'index': 0,
        'sample_score': {
            'sample_id': 0,
            'sample_metadata': {
                'task_id': 'task-1',
                'deliverable_files': [{
                    'path': 'deliverable_files/report.txt',
                    'local_path': str(source_file),
                }],
            },
            'score': {
                'prediction': 'Done.',
                'extracted_prediction': 'Done.',
            },
        },
    }
    with open(review_dir / 'gdpval_default.jsonl', 'w', encoding='utf-8') as f:
        f.write(json.dumps(review_item) + '\n')

    adapter._export_submission(report_dir)

    submission_dir = report_dir / 'gdpval_submission'
    assert (submission_dir / 'deliverable_files/task-1/report.txt').read_text(encoding='utf-8') == 'hello'
    assert (submission_dir / 'submission_info.json').is_file()

    import pandas as pd
    table = pd.read_parquet(submission_dir / 'data/train-00000-of-00001.parquet')
    assert table.loc[0, 'deliverable_text'] == 'Done.'
    assert table.loc[0, 'deliverable_files'] == ['deliverable_files/task-1/report.txt']


def test_adapter_requires_parquet_dependencies(monkeypatch: Any) -> None:
    def fake_check_import(**kwargs: Any) -> bool:
        assert kwargs['module_name'] == ['pandas', 'pyarrow']
        assert kwargs['raise_error'] is True
        assert kwargs['feature_name'] == 'GDPval submission export'
        raise ImportError('`pyarrow` not found. Please run `pip install pyarrow` to use GDPval submission export.')

    monkeypatch.setattr('evalscope.benchmarks.gdpval.gdpval_adapter.check_import', fake_check_import)

    with pytest.raises(ImportError, match='pyarrow.*GDPval submission export'):
        make_adapter()


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


class FakeLLMJudge:
    model_id = 'fake-judge'

    def __init__(self) -> None:
        self.prompt = ''

    def judge(self, prompt: str) -> str:
        self.prompt = prompt
        return json.dumps({'score': 0.75, 'explanation': 'Looks mostly correct.'})
