# Copyright (c) Alibaba, Inc. and its affiliates.
"""TORGO reports corpus WER and CER (total edits / total reference words or characters), as jiwer computes them."""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

# One error in a 1-word utterance and a perfect 10-word one: corpus WER is 1 / 11, not mean(1/1, 0).
UTTERANCES = [
    ('yes', 'yet'),
    (
        'the quick brown fox jumps over the lazy sleeping dog',
        'the quick brown fox jumps over the lazy sleeping dog',
    ),
]


def test_cer_and_wer_pool_edits_over_the_reference() -> None:
    jiwer = pytest.importorskip('jiwer')
    adapter = get_benchmark('torgo', TaskConfig(datasets=['torgo'], dataset_args={'torgo': {'metric_list': ['cer', 'wer']}}))
    scores = []
    for index, (reference, hypothesis) in enumerate(UTTERANCES):
        state = TaskState(
            model='mock',
            sample=Sample(input='q', target=reference),
            output=ModelOutput.from_content('mock', hypothesis),
            completed=True,
        )
        score = adapter.calculate_metrics(state).score
        score.value['sem_score'] = float(index)  # added by batch scoring; a similarity, so it stays a plain mean
        scores.append(SampleScore(score=score, sample_id=index))

    results = {agg.metric_name: agg.score for agg in adapter.aggregate_scores(scores)}
    references, hypotheses = zip(*UTTERANCES)
    assert results['wer'] == pytest.approx(1 / 11)
    assert results['wer'] == pytest.approx(jiwer.wer(list(references), list(hypotheses)))
    assert results['cer'] == pytest.approx(jiwer.cer(list(references), list(hypotheses)))
    assert results['sem_score'] == pytest.approx(0.5)
    assert adapter._benchmark_meta.evaluation_version == 'v1.1'
