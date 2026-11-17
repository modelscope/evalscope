"""Unit tests for the VLMs Are Biased adapter and official scoring rules."""
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
from PIL import Image

from evalscope.api.metric import MetricDirection, SampleScore
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.vlms_are_biased.utils import normalize_answer, score_answer
from evalscope.config import TaskConfig
from evalscope.metrics.semantics.catalog import METRIC_DEFINITIONS


def _adapter(**dataset_args: Any) -> Any:
    task_config = TaskConfig(
        model='mock', datasets=['vlms_are_biased'], dataset_args={'vlms_are_biased': dataset_args}
    )
    return get_benchmark('vlms_are_biased', task_config)


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2)).save(buffer, format='PNG')
    return buffer.getvalue()


def _record(**overrides: Any) -> Dict[str, Any]:
    record = {
        'image': {
            'bytes': _image_bytes(),
            'path': 'sample.png'
        },
        'ID': 'logos-shoe-001-fourstripes-notitle-q1-384',
        'legacy_ID': 'sample_Q1_px384',
        'topic': 'Logos',
        'sub_topic': 'Shoe Logos',
        'prompt': 'How many stripes are visible? Answer with a number in curly brackets, e.g., {9}.',
        'ground_truth': '4',
        'expected_bias': '3',
        'with_title': False,
        'type_of_question': 'Q1',
        'pixel': 384,
    }
    record.update(overrides)
    return record


def test_normalize_answer_matches_official_brace_rule() -> None:
    assert normalize_answer('  {Yes}\n') == 'yes'
    assert normalize_answer('{{4}}') == '4'
    assert normalize_answer('The answer is {4}.') == 'the answer is {4}.'


def test_score_answer_exact_and_bias_matches() -> None:
    assert score_answer('{4}', '4', '3') == {'acc': 1.0, 'bias_ratio': 0.0}
    assert score_answer('{3}', '4', '3') == {'acc': 0.0, 'bias_ratio': 1.0}
    assert score_answer('{2}', '4', '3') == {'acc': 0.0, 'bias_ratio': 0.0}
    assert score_answer('{YES}', 'Yes', 'No') == {'acc': 1.0, 'bias_ratio': 0.0}


def test_score_answer_uses_official_digit_fallback() -> None:
    assert score_answer('There are {4} visible stripes.', '4', '3') == {'acc': 1.0, 'bias_ratio': 0.0}
    assert score_answer('I see 3 stripes.', '4', '3') == {'acc': 0.0, 'bias_ratio': 1.0}
    assert score_answer('I counted 3, then 4.', '4', '3') == {'acc': 0.0, 'bias_ratio': 0.0}


def test_original_control_omits_bias_ratio() -> None:
    assert score_answer('{Audi}', 'Audi', None) == {'acc': 1.0}


def test_bias_ratio_semantics_are_lower_is_better() -> None:
    semantics = METRIC_DEFINITIONS['bias_ratio'].resolve('bias_ratio')

    assert semantics.metric_name == 'Bias Ratio'
    assert semantics.direction is MetricDirection.LOWER_IS_BETTER


def test_record_to_sample_preserves_official_prompt_and_metadata() -> None:
    sample = _adapter().record_to_sample(_record())

    assert sample.target == '4'
    assert sample.input[0].content[0].image.startswith('data:image/png;base64,')
    assert sample.input[0].content[1].text == _record()['prompt']
    assert sample.metadata == {
        'id': 'logos-shoe-001-fourstripes-notitle-q1-384',
        'legacy_id': 'sample_Q1_px384',
        'data_version': 'v2',
        'topic': 'Logos',
        'sub_topic': 'Shoe Logos',
        'type_of_question': 'Q1',
        'expected_bias': '3',
        'with_title': False,
        'pixel': 384,
    }


def test_v2_conversion_preserves_revised_illusion_prompt_and_loads_image() -> None:
    import base64

    prompt = 'Are the two red lines equal in length? Answer in curly brackets.'
    sample = _adapter().record_to_sample(
        _record(
            ID='illusion-ponzo-013-diff0p15-notitle-q1-384',
            legacy_ID='ponzo_013_notitle_Q1_px384',
            prompt=prompt,
            ground_truth='No',
            expected_bias='Yes',
            topic='Optical Illusion',
        )
    )

    assert sample.input[0].content[1].text == prompt
    image_bytes = base64.b64decode(sample.input[0].content[0].image.split(',', 1)[1])
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        assert image.size == (2, 2)
    assert sample.metadata['legacy_id'] == 'ponzo_013_notitle_Q1_px384'
    assert sample.target == 'No'


def test_v1_requires_explicit_legacy_selection() -> None:
    record = _record(ID='sample_Q1_px384')
    record.pop('legacy_ID')
    with pytest.raises(ValueError, match='default_subset="v1-legacy"'):
        _adapter().record_to_sample(record)

    legacy = _adapter(default_subset='v1-legacy')
    sample = legacy.record_to_sample(record)
    assert sample.metadata['data_version'] == 'v1-legacy'
    assert sample.metadata['legacy_id'] is None
    assert legacy.pretty_name == 'VLMs Are Biased (v1 legacy)'
    assert legacy.benchmark_meta.evaluation_version == 'v1.0'
    assert _adapter().benchmark_meta.evaluation_version == 'v1.1'


@pytest.mark.parametrize('hub', ['modelscope', 'huggingface'])
def test_data_versions_have_distinct_pinned_cache_identities(hub: str) -> None:
    from evalscope.evaluation_versioning import (
        ResolvedBenchmarkSpec,
        build_benchmark_identity,
        cache_source_for_identity,
    )

    identities = []
    for subset in ['default', 'v1-legacy']:
        config = TaskConfig(
            model='mock',
            datasets=['vlms_are_biased'],
            dataset_hub=hub,
            dataset_args={'vlms_are_biased': {'default_subset': subset}},
        )
        adapter = get_benchmark('vlms_are_biased', config)
        meta = adapter.benchmark_meta
        spec = ResolvedBenchmarkSpec.from_meta(meta, config)
        assert spec.dataset_revision is not None and len(spec.dataset_revision) == 40
        assert spec.dataset_id == (
            'anvo25/vlms-are-biased' if hub == 'huggingface' else 'evalscope/vlms-are-biased'
        )
        identities.append(build_benchmark_identity(spec, meta.evaluation_version, config))

    assert identities[0].fingerprint != identities[1].fingerprint
    assert identities[0].evaluation_version == 'v1.1'
    assert identities[1].evaluation_version == 'v1.0'
    with pytest.raises(ValueError, match='identity does not match'):
        cache_source_for_identity(identities[0], identities[1], False, False)


def test_loader_selects_v2_splits_and_rejects_legacy_run_cache(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from evalscope import run_task
    from evalscope.benchmarks.vlms_are_biased.vlms_are_biased_adapter import SUBSET_LIST

    data = tmp_path / 'data'
    (data / 'v2').mkdir(parents=True)
    configs = []
    for version, prefix in [('default', 'v2/'), ('v1-legacy', '')]:
        files = []
        for subset in SUBSET_LIST:
            record = _record(ID=f'{subset}-new-id', legacy_ID=f'{subset}-old-id')
            if version == 'v1-legacy':
                record['ID'] = record.pop('legacy_ID')
            pq.write_table(pa.Table.from_pylist([record]), data / f'{prefix}{subset}.parquet')
            files.append({'split': subset, 'path': f'{prefix}{subset}.parquet'})
        configs.append({'config_name': version, 'data_files': files})
    (data / 'README.md').write_text(f'---\n{yaml.safe_dump({"configs": configs})}---\n', encoding='utf-8')

    adapter = _adapter(local_path=str(data))
    loaded = adapter.load_dataset()
    assert list(loaded.keys()) == SUBSET_LIST
    assert [loaded[subset][0].metadata['id'] for subset in SUBSET_LIST] == [
        f'{subset}-new-id' for subset in SUBSET_LIST
    ]
    assert all(loaded[subset][0].metadata['data_version'] == 'v2' for subset in SUBSET_LIST)

    work = tmp_path / 'run'
    options = dict(
        model='offline', eval_type='mock_llm', datasets=['vlms_are_biased'],
        no_timestamp=True, work_dir=str(work), collect_perf=False,
    )
    run_task(TaskConfig(
        **options,
        dataset_args={'vlms_are_biased': {
            'local_path': str(data), 'default_subset': 'v1-legacy', 'subset_list': ['main'],
        }},
    ))
    with pytest.raises(ValueError, match='identity does not match'):
        run_task(TaskConfig(
            **options, use_cache=str(work),
            dataset_args={'vlms_are_biased': {'local_path': str(data), 'subset_list': ['main']}},
        ))


def test_match_score_uses_expected_bias_metadata() -> None:
    score = _adapter().match_score(
        original_prediction='{3}',
        filtered_prediction='3',
        reference='4',
        task_state=SimpleNamespace(metadata={'expected_bias': '3'}),
    )

    assert score.value == {'acc': 0.0, 'bias_ratio': 1.0}
    assert score.main_score_name == 'acc'


def test_aggregation_reports_official_per_topic_accuracy() -> None:
    sample_scores = [
        SampleScore(sample_id='1', score=Score(value={'acc': 1.0, 'bias_ratio': 0.0}), sample_metadata={'topic': 'Logos'}),
        SampleScore(sample_id='2', score=Score(value={'acc': 0.0, 'bias_ratio': 1.0}), sample_metadata={'topic': 'Logos'}),
        SampleScore(sample_id='3', score=Score(value={'acc': 1.0, 'bias_ratio': 0.0}), sample_metadata={'topic': 'Flags'}),
    ]

    aggregate_scores = _adapter().aggregate_scores(sample_scores)
    by_identity = {(score.metric_name, score.dimensions.get('topic')): score.score for score in aggregate_scores}

    assert by_identity[('accuracy', None)] == 2 / 3
    assert by_identity[('bias_ratio', None)] == 1 / 3
    assert by_identity[('accuracy', 'Logos')] == 0.5
    assert by_identity[('accuracy', 'Flags')] == 1.0
