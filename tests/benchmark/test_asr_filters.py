import pytest
from collections import OrderedDict
from typing import Any, Type

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.benchmarks.common_voice_15.common_voice_15_adapter import CommonVoice15Adapter
from evalscope.benchmarks.fleurs.fleurs_adapter import FLEURSAdapter
from evalscope.benchmarks.librispeech.librispeech_adapter import LibriSpeechAdapter
from evalscope.benchmarks.torgo.torgo_adapter import TorgoAdapter
from evalscope.config import TaskConfig
from evalscope.filters import extraction  # noqa: F401  # registered filters


def make_adapter(adapter_cls: Type[Any], name: str, metric_list: list[str]) -> Any:
    meta = BenchmarkMeta(
        name=name,
        dataset_id='mock',
        filters=OrderedDict({'remove_until': '<asr_text>'}),
        metric_list=metric_list,
    )
    cfg = TaskConfig(datasets=[name])
    return adapter_cls(benchmark_meta=meta, task_config=cfg)


@pytest.mark.parametrize(
    ('adapter_cls', 'name', 'metric_list', 'metadata', 'expected_metrics'),
    [
        (LibriSpeechAdapter, 'librispeech', ['wer'], {}, {'wer': 0.0}),
        (FLEURSAdapter, 'fleurs', ['wer'], {'lang_id': 'en'}, {'wer': 0.0}),
        (CommonVoice15Adapter, 'common_voice_15', ['wer'], {'locale': 'en'}, {'wer': 0.0}),
    ],
)
def test_asr_match_score_uses_filtered_prediction(
    adapter_cls: Type[Any],
    name: str,
    metric_list: list[str],
    metadata: dict[str, str],
    expected_metrics: dict[str, float],
) -> None:
    adapter = make_adapter(adapter_cls, name, metric_list)
    original_prediction = 'language English<asr_text>hello world'
    sample = Sample(input='audio', target='hello world', metadata=metadata)
    task_state = TaskState(model='mock', sample=sample, completed=True)

    filtered_prediction = adapter.filter_prediction(original_prediction, task_state)
    score = adapter.match_score(original_prediction, filtered_prediction, sample.target, task_state)

    assert filtered_prediction == 'hello world'
    assert score.prediction == original_prediction
    assert score.extracted_prediction == 'hello world'
    assert score.value == expected_metrics


def test_torgo_match_score_uses_filtered_prediction() -> None:
    adapter = make_adapter(TorgoAdapter, 'torgo', [])

    def exact_match_score(reference: str, prediction: str) -> float:
        return 0.0 if prediction == 'hello world' else 1.0

    adapter.jiwer_cer = exact_match_score
    adapter.jiwer_wer = exact_match_score
    original_prediction = 'language English<asr_text>hello world'
    sample = Sample(input='audio', target='hello world')
    task_state = TaskState(model='mock', sample=sample, completed=True)

    filtered_prediction = adapter.filter_prediction(original_prediction, task_state)
    score = adapter.match_score(original_prediction, filtered_prediction, sample.target, task_state)

    assert filtered_prediction == 'hello world'
    assert score.prediction == original_prediction
    assert score.extracted_prediction == 'hello world'
    assert score.value == {'cer': 0.0, 'wer': 0.0}


def test_asr_match_score_handles_none_filtered_prediction() -> None:
    adapter = make_adapter(LibriSpeechAdapter, 'librispeech', ['wer'])
    sample = Sample(input='audio', target='hello world')
    task_state = TaskState(model='mock', sample=sample, completed=True)

    score = adapter.match_score('language English<asr_text>hello world', None, sample.target, task_state)

    assert score.extracted_prediction == ''
    assert score.value == {'wer': 1.0}
