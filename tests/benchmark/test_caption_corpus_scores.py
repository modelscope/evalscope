# Copyright (c) Alibaba, Inc. and its affiliates.
"""MSR-VTT and MSVD report corpus-level COCO caption metrics, whatever eval_batch_size windows scoring used."""
import json

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.caption import metrics as caption_metrics
from evalscope.config import TaskConfig

PREDICTIONS = ['a man is playing a guitar', 'a woman is cooking in a kitchen', 'a dog runs on the grass']
REFERENCES = [
    ['a man plays the guitar', 'someone is playing a guitar on stage'],
    ['a woman cooks food in the kitchen', 'a lady is preparing a meal'],
    ['a dog is running across a field', 'a puppy runs on grass'],
]


@pytest.mark.parametrize('name', ['msr_vtt', 'msvd'])
def test_aggregate_uses_one_corpus_not_scoring_windows(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('pycocoevalcap.cider.cider')
    monkeypatch.setattr(caption_metrics.shutil, 'which', lambda _: None)  # same tokenizer with or without Java
    adapter = get_benchmark(name, TaskConfig(datasets=[name]))

    scores = []
    for index, (prediction, references) in enumerate(zip(PREDICTIONS, REFERENCES)):
        state = TaskState(
            model='mock',
            sample=Sample(input='q', target=json.dumps(references), metadata={'references': references}),
            output=ModelOutput.from_content('mock', prediction),
            completed=True,
        )
        score = SampleScore(
            score=adapter.match_score(prediction, prediction, state.target, state),
            sample_id=index,
            sample_metadata=state.metadata,
        )
        adapter.batch_calculate_metrics([state], [score])  # eval_batch_size=1: one caption per window
        scores.append(score)

    whole_corpus = caption_metrics.compute_caption_scores(PREDICTIONS, REFERENCES)
    expected_cider = sum(values['CIDEr'] for values in whole_corpus) / len(whole_corpus)
    assert expected_cider > 0
    from pycocoevalcap.bleu.bleu import Bleu

    tokenized_refs = caption_metrics._simple_caption_tokenize(
        {i: [{'caption': ref} for ref in refs] for i, refs in enumerate(REFERENCES)}
    )
    tokenized_preds = caption_metrics._simple_caption_tokenize(
        {i: [{'caption': pred}] for i, pred in enumerate(PREDICTIONS)}
    )
    corpus_bleu, _ = Bleu(4).compute_score(tokenized_refs, tokenized_preds, verbose=0)

    aggregates = {(agg.metric_name, agg.dimensions.get('ngram')): agg.score for agg in adapter.aggregate_scores(scores)}
    assert aggregates[('cider', None)] == pytest.approx(expected_cider)
    assert aggregates[('bleu', 4)] == pytest.approx(corpus_bleu[3])
    assert adapter._benchmark_meta.evaluation_version == 'v1.1'
