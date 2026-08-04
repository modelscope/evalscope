import hashlib
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.omnidoc_bench.v1_6 import omnidoc_bench_v1_6_adapter as v16
from evalscope.benchmarks.omnidoc_bench.v1_6.sandbox_scorer import (
    RESULT_SENTINEL,
    build_scoring_program,
    parse_scoring_result,
)
from evalscope.config import TaskConfig
from evalscope.run import run_task


def make_records() -> list[dict]:
    records = []
    subset_counts = (
        ('v1.5', 1355),
        ('equation_hard', 100),
        ('layout_hard', 99),
        ('table_hard', 97),
    )
    for subset, count in subset_counts:
        for _ in range(count):
            index = len(records)
            records.append({
                'layout_dets': [],
                'page_info': {
                    'image_path': f'page_{index}.png',
                    'page_attribute': {
                        'subset': subset
                    },
                },
                'extra': {},
            })
    return records


def write_dataset(root: Path, records: list[dict], image_count: int = 1) -> str:
    annotation = json.dumps(records, ensure_ascii=False).encode('utf-8')
    (root / 'OmniDocBench.json').write_bytes(annotation)
    image_dir = root / 'images'
    image_dir.mkdir()
    for index in range(image_count):
        (image_dir / f'page_{index}.png').write_bytes(b'image')
    return hashlib.sha256(annotation).hexdigest()


def make_adapter(
    root: Path | None = None,
    *,
    limit: int | None = None,
    repeats: int = 1,
    shuffle: bool = False,
    seed: int = 42,
    sandbox: dict | None = None,
) -> v16.OmniDocBenchV16Adapter:
    dataset_args = {'omni_doc_bench_v1_6': {'shuffle': shuffle}}
    if root:
        dataset_args['omni_doc_bench_v1_6']['local_path'] = str(root)
    config = TaskConfig(
        datasets=['omni_doc_bench_v1_6'],
        dataset_args=dataset_args,
        limit=limit,
        repeats=repeats,
        seed=seed,
        sandbox=sandbox,
    )
    return get_benchmark('omni_doc_bench_v1_6', config)


def test_omnidoc_bench_versions_are_registered_separately() -> None:
    legacy = get_benchmark('omni_doc_bench', TaskConfig(datasets=['omni_doc_bench']))
    current = make_adapter()

    assert legacy.dataset_id == 'evalscope/OmniDocBench_tsv'
    assert legacy.eval_split == 'train'
    assert current.dataset_id == 'OpenDataLab/OmniDocBench'
    assert current.eval_split == 'test'
    assert current.sandbox_config == v16.DEFAULT_SANDBOX_CONFIG


def test_v16_local_load_validates_and_selects_before_reading_images(tmp_path: Path, monkeypatch) -> None:
    records = make_records()
    digest = write_dataset(tmp_path, records, image_count=1)
    monkeypatch.setattr(v16, 'ANNOTATION_SHA256', digest)
    adapter = make_adapter(tmp_path, limit=1, repeats=2)

    datasets, fewshot = adapter.load()

    assert fewshot is None
    assert len(datasets['default']) == 2
    assert datasets['default'][0].group_id == datasets['default'][1].group_id
    sample = datasets['default'][0]
    assert [content.type for content in sample.input[0].content] == ['image', 'text']
    assert sample.metadata['image_name'] == 'page_0.png'
    assert sample.metadata['annotation'] == records[0]
    assert sample.metadata['omnidocbench_version'] == 'v1.6'


def test_v16_shuffle_is_deterministic_before_limit(tmp_path: Path, monkeypatch) -> None:
    records = make_records()
    digest = write_dataset(tmp_path, records, image_count=11)
    monkeypatch.setattr(v16, 'ANNOTATION_SHA256', digest)

    datasets, _ = make_adapter(tmp_path, limit=1, shuffle=True, seed=1238).load()

    assert datasets['default'][0].metadata['image_name'] == 'page_10.png'


def test_v16_rejects_wrong_digest(tmp_path: Path) -> None:
    write_dataset(tmp_path, make_records())
    adapter = make_adapter(tmp_path, limit=1)

    with pytest.raises(ValueError, match='supports v1.6 only'):
        adapter.load()


def test_v16_rejects_invalid_schema_count_and_path(tmp_path: Path, monkeypatch) -> None:
    adapter = make_adapter(tmp_path)
    cases = [
        ([{'layout_dets': [], 'page_info': {}, 'extra': {}}], 'sample count'),
        ([{}] * v16.EXPECTED_SAMPLE_COUNT, 'must contain layout_dets'),
    ]
    for index, (records, message) in enumerate(cases):
        path = tmp_path / f'invalid_{index}.json'
        content = json.dumps(records).encode('utf-8')
        path.write_bytes(content)
        monkeypatch.setattr(v16, 'ANNOTATION_SHA256', hashlib.sha256(content).hexdigest())
        with pytest.raises(ValueError, match=message):
            adapter._load_and_validate_annotation(path)

    records = make_records()
    records[0]['page_info']['image_path'] = '../escape.png'
    path = tmp_path / 'invalid_path.json'
    content = json.dumps(records).encode('utf-8')
    path.write_bytes(content)
    monkeypatch.setattr(v16, 'ANNOTATION_SHA256', hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match='Invalid OmniDocBench v1.6 image path'):
        adapter._load_and_validate_annotation(path)

    records[0]['page_info']['image_path'] = '..\\escape.png'
    content = json.dumps(records).encode('utf-8')
    path.write_bytes(content)
    monkeypatch.setattr(v16, 'ANNOTATION_SHA256', hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match='Invalid OmniDocBench v1.6 image path'):
        adapter._load_and_validate_annotation(path)


def test_v16_rejects_invalid_subset_counts_and_missing_selected_image(tmp_path: Path, monkeypatch) -> None:
    records = make_records()
    records[0]['page_info']['page_attribute']['subset'] = 'other'
    digest = write_dataset(tmp_path, records, image_count=0)
    monkeypatch.setattr(v16, 'ANNOTATION_SHA256', digest)
    with pytest.raises(ValueError, match='subset counts'):
        make_adapter(tmp_path, limit=1).load()

    records[0]['page_info']['page_attribute']['subset'] = 'v1.5'
    content = json.dumps(records).encode('utf-8')
    (tmp_path / 'OmniDocBench.json').write_bytes(content)
    monkeypatch.setattr(v16, 'ANNOTATION_SHA256', hashlib.sha256(content).hexdigest())
    with pytest.raises(FileNotFoundError, match='page_0.png'):
        make_adapter(tmp_path, limit=1).load()


def test_v16_remote_loader_pins_modelscope_revision(tmp_path: Path) -> None:
    annotation_path = tmp_path / 'OmniDocBench.json'
    annotation_path.write_text('[]', encoding='utf-8')
    with patch.object(v16, 'download_dataset_file', return_value=str(annotation_path)) as download_file:
        adapter = make_adapter()
        with patch.object(adapter, '_load_and_validate_annotation', return_value=[]):
            dataset = adapter.load_subset('default', v16.DictDataLoader)

    assert len(dataset) == 0
    assert download_file.call_args.kwargs['data_id_or_path'] == 'OpenDataLab/OmniDocBench'
    assert download_file.call_args.kwargs['file_path'] == 'OmniDocBench.json'
    assert download_file.call_args.kwargs['revision'] == v16.DATASET_REVISION


def test_v16_scores_each_page_with_one_sandbox_call() -> None:
    adapter = make_adapter(sandbox={'enabled': True})
    metrics = {
        'text_block_Edit_dist': 0.2,
        'display_formula_CDM': 80.0,
        'table_TEDS': 70.0,
    }
    sample = Sample(
        input='question',
        target='',
        metadata={
            'annotation': {
                'layout_dets': [],
                'page_info': {
                    'image_path': 'page.png'
                },
                'extra': {},
            },
            'image_name': 'page.png',
        },
    )
    task_state = TaskState(model='model', sample=sample)
    result = {'status': 'success', 'output': f'official logs\n{RESULT_SENTINEL}{json.dumps(metrics)}\n'}

    with patch.object(adapter, 'execute_code_in_sandbox', return_value=result) as execute:
        score = adapter.match_score('# page', '# page', '', task_state)

    execute.assert_called_once()
    assert score.value == metrics
    assert score.metadata['official_scorer_commit'] == v16.OFFICIAL_SCORER_COMMIT


def test_v16_scoring_program_is_valid_python() -> None:
    program = build_scoring_program(
        {
            'layout_dets': [],
            'page_info': {
                'image_path': 'page.png'
            },
            'extra': {},
        },
        'page.png',
        '# markdown',
    )

    compile(program, '<omnidocbench-v1.6-sandbox>', 'exec')
    assert '"match_method": "quick_match"' in program
    assert '(work_dir / "result").mkdir()' in program


def test_v16_scoring_errors_are_explicit() -> None:
    with pytest.raises(RuntimeError, match='did not return a metric result'):
        parse_scoring_result({'status': 'success', 'output': 'official logs'})
    with pytest.raises(RuntimeError, match='scoring failed'):
        parse_scoring_result({'status': 'timeout', 'error': 'timeout'})
    invalid_metric = RESULT_SENTINEL + json.dumps({'table_TEDS': 101})
    with pytest.raises(RuntimeError, match='expected 0-100 range'):
        parse_scoring_result({'status': 'success', 'output': invalid_metric})


def test_v16_aggregates_page_metrics_then_computes_overall() -> None:
    adapter = make_adapter()
    sample_scores = [
        SampleScore(
            sample_id=0,
            score=Score(value={
                'text_block_Edit_dist': 0.1,
                'display_formula_CDM': 90.0,
            }),
        ),
        SampleScore(
            sample_id=1,
            score=Score(value={
                'text_block_Edit_dist': 0.3,
                'table_TEDS': 60.0,
            }),
        ),
    ]

    aggregate = {score.metric_name: score for score in adapter.aggregate_scores(sample_scores)}

    assert aggregate['text_block_Edit_dist'].score == pytest.approx(0.2)
    assert aggregate['text_block_Edit_dist'].num == 2
    assert aggregate['display_formula_CDM'].num == 1
    assert aggregate['table_TEDS'].num == 1
    assert aggregate['overall'].score == pytest.approx(((1 - 0.2) * 100 + 90 + 60) / 3)
    assert aggregate['overall'].metadata['component_page_denominators'] == {
        'text_block_Edit_dist': 2,
        'display_formula_CDM': 1,
        'table_TEDS': 1,
    }


def test_v16_omits_overall_when_a_component_is_missing() -> None:
    adapter = make_adapter()
    scores = [SampleScore(sample_id=0, score=Score(value={'text_block_Edit_dist': 0.1}))]

    aggregate = adapter.aggregate_scores(scores)

    assert [score.metric_name for score in aggregate] == ['text_block_Edit_dist']


def test_v16_mock_end_to_end_writes_review_and_report(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / 'dataset'
    dataset_root.mkdir()
    digest = write_dataset(dataset_root, make_records(), image_count=1)
    monkeypatch.setattr(v16, 'ANNOTATION_SHA256', digest)
    result = {
        'status': 'success',
        'output': RESULT_SENTINEL + json.dumps({
            'text_block_Edit_dist': 0.2,
            'display_formula_Edit_dist': 0.1,
            'display_formula_CDM': 80.0,
            'table_TEDS': 70.0,
            'table_TEDS_structure_only': 75.0,
            'table_Edit_dist': 0.3,
            'reading_order_Edit_dist': 0.1,
        }),
    }
    output_dir = tmp_path / 'outputs'
    config = TaskConfig(
        model='mock_llm',
        eval_type='mock_llm',
        datasets=['omni_doc_bench_v1_6'],
        dataset_args={'omni_doc_bench_v1_6': {'local_path': str(dataset_root)}},
        sandbox={'enabled': True},
        limit=1,
        work_dir=str(output_dir),
        no_timestamp=True,
    )

    with patch.object(v16.OmniDocBenchV16Adapter, 'execute_code_in_sandbox', return_value=result) as execute:
        run_task(config)

    execute.assert_called_once()
    assert list((output_dir / 'predictions').rglob('*.jsonl'))
    assert list((output_dir / 'reviews').rglob('*.jsonl'))
    assert list((output_dir / 'reports').rglob('*.json'))
