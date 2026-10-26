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


@pytest.mark.parametrize(
    ('utterances', 'expected_wer'),
    [
        (UTTERANCES, 1 / 11),
        ([('', 'hello'), ('yes', 'yes')], 1.0),
        ([('uh', 'hello'), ('yes', 'yes')], 1.0),
        ([('', 'hello')], 1.0),
        ([('', '')], 0.0),
        ([('', ''), ('yes', 'yes')], 0.0),
    ],
    ids=['ordinary', 'empty-reference', 'normalized-empty-reference', 'all-empty', 'silence', 'mixed-silence'],
)
def test_cer_and_wer_pool_edits_over_the_reference(utterances: list[tuple[str, str]], expected_wer: float) -> None:
    jiwer = pytest.importorskip('jiwer')
    from evalscope.metrics.utils.text_normalizer.wer import normalize_text

    adapter = get_benchmark('torgo', TaskConfig(datasets=['torgo'], dataset_args={'torgo': {'metric_list': ['cer', 'wer']}}))
    scores = []
    for index, (reference, hypothesis) in enumerate(utterances):
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
    references = [normalize_text(reference, 'en') for reference, _ in utterances]
    hypotheses = [normalize_text(hypothesis, 'en') for _, hypothesis in utterances]
    assert results['wer'] == pytest.approx(expected_wer)
    assert results['wer'] == pytest.approx(jiwer.wer(list(references), list(hypotheses)))
    assert results['cer'] == pytest.approx(jiwer.cer(list(references), list(hypotheses)))
    assert results['sem_score'] == pytest.approx((len(utterances) - 1) / 2)
    assert adapter._benchmark_meta.evaluation_version == 'v1.1'
