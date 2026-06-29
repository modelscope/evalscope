import asyncio
import base64
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

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
from evalscope.benchmarks.gdpval.gdpval_scorer import GDPvalLocalScorer, parse_rubric_items, read_xlsx_workbook
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
        metric_list=['submission_ready'],
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


def test_parse_rubric_items_accepts_gdpval_json_string() -> None:
    items = parse_rubric_items(
        json.dumps([{
            'score': 2,
            'criterion': 'The workbook contains a worksheet named exactly Sample Size Calculation.',
            'rubric_item_id': 'item-1',
            'tags': ['true'],
        }])
    )

    assert len(items) == 1
    assert items[0].score == 2
    assert items[0].rubric_item_id == 'item-1'


def test_read_xlsx_workbook_handles_empty_value_node(tmp_path: Path) -> None:
    path = tmp_path / 'empty-value.xlsx'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('[Content_Types].xml', _content_types(1))
        archive.writestr('_rels/.rels', _root_rels())
        archive.writestr('xl/workbook.xml', _workbook_xml(['Sheet1']))
        archive.writestr('xl/_rels/workbook.xml.rels', _workbook_rels(1))
        archive.writestr(
            'xl/worksheets/sheet1.xml',
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<sheetData><row r="1"><c r="A1"><v/></c></row></sheetData></worksheet>'
            ),
        )

    workbook = read_xlsx_workbook(path)

    assert workbook is not None
    assert workbook.first_sheet is not None
    assert workbook.first_sheet.rows == [['']]


def test_local_rubric_scorer_scores_checkable_spreadsheet_items(tmp_path: Path) -> None:
    reference_path = tmp_path / 'Population.xlsx'
    sample_path = tmp_path / 'Sample.xlsx'
    population_headers = [
        'No', 'Division', 'Sub-Division', 'Country', 'Legal Entity', 'KRIs', 'Q3 2024 KRI', 'Q2 2024 KRI'
    ]
    _write_xlsx(
        reference_path,
        {
            'Population': [
                population_headers,
                [1, 'Corporate Bank', 'Corporate Loans', 'Italy', 'Willett Bank Rome', 'Total clients', 120, 100],
                [2, 'Retail Bank', 'EMEA', 'UAE', 'Willett Bank UAE', 'HR Clients', 0, 0],
                [3, 'Corporate Bank', 'Corporate Loans', 'Pakistan', 'Willett Bank Pakistan', 'MR Clients', 12, 10],
            ]
        },
    )
    _write_xlsx(
        sample_path,
        {
            'Sample': [
                population_headers + ['Variance (%)', 'Sample'],
                [1, 'Corporate Bank', 'Corporate Loans', 'Italy', 'Willett Bank Rome', 'Total clients', 120, 100, 20, 1],
                [2, 'Retail Bank', 'EMEA', 'UAE', 'Willett Bank UAE', 'HR Clients', 0, 0, 0, 1],
            ],
            'Sample Size Calculation': [
                ['Calculation', 'Value'],
                ['Confidence Level', '90%'],
                ['Tolerable Error Rate', '10%'],
                ['Population Size (N)', 3],
                ['Formula', 'z = 1.645, p = 0.5, e = 0.10, finite population correction'],
                ['Final Sample Size', 2],
            ],
        },
    )
    rubric_json = json.dumps([
        {
            'score': 2,
            'criterion': "The submitted deliverable is an Excel workbook file whose basename is 'Sample'.",
            'rubric_item_id': 'sample-file',
        },
        {
            'score': 2,
            'criterion': "The workbook contains a worksheet named exactly 'Sample Size Calculation'.",
            'rubric_item_id': 'calc-sheet',
        },
        {
            'score': 2,
            'criterion': "The 'Sample Size Calculation' worksheet shows the population size N used.",
            'rubric_item_id': 'population-size',
        },
        {
            'score': 1,
            'criterion': "If 'Pakistan' occurs in the Country column in the Population reference, at least one such row is flagged as sampled.",
            'rubric_item_id': 'pakistan',
        },
    ])
    metadata = {
        'rubric_json': rubric_json,
        'host_reference_files': [str(reference_path)],
        'deliverable_files': [{
            'path': 'deliverable_files/Sample.xlsx',
            'local_path': str(sample_path),
        }],
    }

    score = GDPvalLocalScorer(metadata, 'Done.').score()
    results = {result.rubric_item_id: result for result in score.item_results}

    assert score.total_items == 4
    assert score.evaluated_items == 4
    assert results['sample-file'].criteria_met is True
    assert results['calc-sheet'].criteria_met is True
    assert results['population-size'].criteria_met is True
    assert results['pakistan'].criteria_met is False
    assert score.score == 6 / 7


def test_match_score_runs_local_rubric_scorer(tmp_path: Path) -> None:
    reference_path = tmp_path / 'Population.xlsx'
    sample_path = tmp_path / 'Sample.xlsx'
    _write_xlsx(reference_path, {'Population': [['No', 'Country'], [1, 'Pakistan']]})
    _write_xlsx(sample_path, {'Sample': [['No', 'Country', 'Sample'], [1, 'Pakistan', 1]]})
    adapter = make_adapter(scoring_mode='local_rubric_judge')
    sample = Sample(
        input='prompt',
        target='',
        metadata={
            'rubric_json': json.dumps([{
                'score': 1,
                'criterion': (
                    "If 'Pakistan' occurs in the Country column in the Population reference, at least one such row "
                    'is flagged as sampled.'
                ),
                'rubric_item_id': 'pakistan',
            }]),
            'host_reference_files': [str(reference_path)],
            'deliverable_files': [{
                'path': 'deliverable_files/Sample.xlsx',
                'local_path': str(sample_path),
            }],
        },
    )
    state = TaskState(model='mock', sample=sample, completed=True)

    score = adapter.match_score('Done.', 'Done.', '', state)

    assert score.main_score_name == 'local_rubric_score'
    assert score.value['submission_ready'] == 1.0
    assert score.value['local_rubric_score'] == 1.0
    assert score.metadata['local_rubric_score_note'].startswith('EvalScope local deterministic rubric score')


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
    adapter = make_adapter(scoring_mode='openai_auto_grader_submission')
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


def _write_xlsx(path: Path, sheets: Dict[str, List[List[Any]]]) -> None:
    sheet_names = list(sheets.keys())
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('[Content_Types].xml', _content_types(len(sheet_names)))
        archive.writestr('_rels/.rels', _root_rels())
        archive.writestr('xl/workbook.xml', _workbook_xml(sheet_names))
        archive.writestr('xl/_rels/workbook.xml.rels', _workbook_rels(len(sheet_names)))
        for idx, (_, rows) in enumerate(sheets.items(), start=1):
            archive.writestr(f'xl/worksheets/sheet{idx}.xml', _sheet_xml(rows))


def _content_types(sheet_count: int) -> str:
    sheet_overrides = ''.join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f'{sheet_overrides}</Types>'
    )


def _root_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def _workbook_xml(sheet_names: List[str]) -> str:
    sheets_xml = ''.join(
        f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheets_xml}</sheets></workbook>'
    )


def _workbook_rels(sheet_count: int) -> str:
    rels = ''.join(
        f'<Relationship Id="rId{idx}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{rels}</Relationships>'
    )


def _sheet_xml(rows: List[List[Any]]) -> str:
    rows_xml = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            ref = f'{_column_name(col_idx)}{row_idx}'
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        rows_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(rows_xml)}</sheetData></worksheet>'
    )


def _column_name(index: int) -> str:
    name = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord('A') + remainder) + name
    return name
