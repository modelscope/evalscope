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
    from evalscope.metrics.math import runtime
    assert runtime._pool is None
