# Copyright (c) Alibaba, Inc. and its affiliates.
"""ASR benchmarks report corpus WER (total errors / total reference words), as lmms-eval and Qwen-Audio do."""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

# One error in a 2-word utterance and a perfect 18-word utterance: corpus WER is 1 / 20, not mean(1/2, 0).
UTTERANCES = [
    ('thank you', 'thank yo'),
    (
        'he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces',
        'he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces',
    ),
]
CASES = {
    'librispeech': ({}, 'v1.2'),
    'fleurs': ({'lang_id': 'en'}, 'v1.1'),
    'common_voice_15': ({'locale': 'en'}, 'v1.1'),
}


@pytest.mark.parametrize('name', list(CASES))
def test_wer_pools_errors_over_reference_words(name: str) -> None:
    metadata, version = CASES[name]
    adapter = get_benchmark(name, TaskConfig(datasets=[name]))
    scores = []
    for index, (reference, hypothesis) in enumerate(UTTERANCES):
        state = TaskState(
            model='mock',
            sample=Sample(input='q', target=reference, metadata=dict(metadata)),
            output=ModelOutput.from_content('mock', hypothesis),
            completed=True,
        )
        scores.append(SampleScore(score=adapter.calculate_metrics(state).score, sample_id=index))

    (wer,) = adapter.aggregate_scores(scores)
    assert wer.metric_name == 'wer'
    assert wer.score == pytest.approx(1 / 20)
    assert adapter._benchmark_meta.evaluation_version == version
