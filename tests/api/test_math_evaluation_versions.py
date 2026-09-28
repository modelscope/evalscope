"""A scoring migration invalidates reviews while retaining explicit prediction replay."""

import json
from typing import Any

import pytest

from evalscope import TaskConfig, run_task
from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
from evalscope.models.mockllm import MockLLM

AFFECTED_BENCHMARKS = (
    'agieval',
    'aime24',
    'aime25',
    'aime26',
    'amc',
    'arxivmath',
    'chartqa',
    'cmath',
    'cmmu',
    'competition_math',
    'docmath',
    'gsm8k',
    'gsm8k_indic',
    'gsm8k_v',
    'hipho',
    'hmmt25',
    'hmmt26',
    'hmmt_nov25',
    'imo_answerbench',
    'math_500',
    'math_verse',
    'math_vision',
    'math_vista',
    'measure_bench',
    'mgsm',
    'minerva_math',
    'olympiad_bench',
    'poly_math',
    'tir_bench',
)


@pytest.mark.parametrize('benchmark', AFFECTED_BENCHMARKS)
def test_every_affected_benchmark_declares_the_next_minor_version(benchmark: str) -> None:
    expected_version = 'v1.2' if benchmark == 'olympiad_bench' else 'v1.1'
    assert get_benchmark(benchmark).benchmark_meta.evaluation_version == expected_version


def test_gsm8k_old_review_is_blocked_and_predictions_can_be_rescored(tmp_path: Any, monkeypatch: Any) -> None:
    data = tmp_path / 'data'
    data.mkdir()
    (data/'test.jsonl').write_text(json.dumps({'question':'A fixed actual-style word problem',
                                             'answer':'Reasoning. #### 18'})+'\n')
    base = dict(model='offline',eval_type='mock_llm',datasets=['gsm8k'],no_timestamp=True,
                work_dir=str(tmp_path/'run'),judge={'strategy':'rule'},
                dataset_args={'gsm8k':{'local_path':str(data),'few_shot_num':0,'subset_list':['default']}})
    get_benchmark('gsm8k')
    metadata = BENCHMARK_REGISTRY['gsm8k']
    monkeypatch.setattr(MockLLM,'default_output',r'\boxed{18}')
    monkeypatch.setattr(metadata,'evaluation_version','v1.0')
    run_task(TaskConfig(**base))
    prediction = tmp_path/'run/predictions/offline/gsm8k_default.jsonl'
    previous = prediction.read_bytes()
    monkeypatch.setattr(metadata,'evaluation_version','v1.1')
    with pytest.raises(ValueError,match='rerun_review=True'):
        run_task(TaskConfig(**base,use_cache=str(tmp_path/'run')))
    monkeypatch.setattr(MockLLM,'generate',lambda *args,**kwargs: pytest.fail('Prediction reuse must not infer again'))
    run_task(TaskConfig(**base,use_cache=str(tmp_path/'run'),rerun_review=True))
    assert prediction.read_bytes() == previous
    review = json.loads((tmp_path/'run/reviews/offline/gsm8k_default.jsonl').read_text().strip())
    assert review['sample_score']['score']['value']['accuracy'] == 1


@pytest.mark.parametrize('benchmark', ['chartqa', 'measure_bench'])
def test_instrument_old_review_is_blocked_and_predictions_can_be_rescored(
    benchmark: str, tmp_path: Any, monkeypatch: Any
) -> None:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    image = io.BytesIO()
    Image.new('RGB', (8, 8)).save(image, format='PNG')
    if benchmark == 'chartqa':
        record = {'image': {'bytes': image.getvalue()}, 'question': 'Read the chart.',
                  'answer': '0', 'type': 'human_test'}
        subset, prediction, metric = 'human_test', 'ANSWER: 0.0', 'relaxed_acc'
    else:
        record = {'image': {'bytes': image.getvalue()}, 'question': 'Read the instrument.',
                  'question_id': 'fraction-meter', 'image_type': 'ruler', 'design': 'linear',
                  'evaluator': 'interval_matching',
                  'evaluator_kwargs': '{"interval": [0.74, 0.76], "units": ["m"]}'}
        subset, prediction, metric = 'real_world', r'\boxed{\frac{3}{4}} m', 'acc'
    data = tmp_path / 'data'
    data.mkdir()
    pq.write_table(pa.Table.from_pylist([record]), data / 'data.parquet')
    config_name = subset if benchmark == 'chartqa' else 'default'
    split = 'test' if benchmark == 'chartqa' else subset
    (data / 'README.md').write_text(
        f'---\nconfigs:\n- config_name: {config_name}\n  data_files:\n  - split: {split}\n'
        '    path: data.parquet\n---\n'
    )
    work = tmp_path / 'run'
    base = dict(model='offline', eval_type='mock_llm', datasets=[benchmark], no_timestamp=True,
                work_dir=str(work), judge={'strategy': 'rule'},
                dataset_args={benchmark: {'local_path': str(data), 'subset_list': [subset]}})
    get_benchmark(benchmark)
    meta = BENCHMARK_REGISTRY[benchmark]
    monkeypatch.setattr(MockLLM, 'default_output', prediction)
    monkeypatch.setattr(meta, 'evaluation_version', 'v1.0')
    run_task(TaskConfig(**base))
    prediction_path = work / f'predictions/offline/{benchmark}_{subset}.jsonl'
    previous = prediction_path.read_bytes()
    monkeypatch.setattr(meta, 'evaluation_version', 'v1.1')
    with pytest.raises(ValueError, match='rerun_review=True'):
        run_task(TaskConfig(**base, use_cache=str(work)))
    monkeypatch.setattr(MockLLM, 'generate', lambda *args, **kwargs: pytest.fail('Must reuse saved predictions'))
    run_task(TaskConfig(**base, use_cache=str(work), rerun_review=True))
    assert prediction_path.read_bytes() == previous
    review = json.loads((work / f'reviews/offline/{benchmark}_{subset}.jsonl').read_text().strip())
    assert review['sample_score']['score']['value'][metric] == 1
