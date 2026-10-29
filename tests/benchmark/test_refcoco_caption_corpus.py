# Copyright (c) Alibaba, Inc. and its affiliates.
"""RefCOCO caption modes report corpus-level COCO caption metrics, as lmms-eval's refcoco aggregation does."""
from typing import Dict, List, Tuple

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

PREDICTIONS = ['man in red shirt', 'woman on left', 'dog near the bench']
REFERENCES = [
    ['man in red shirt', 'guy wearing red'],
    ['woman on the left', 'left lady'],
    ['brown dog by bench', 'dog'],
]


class _WhitespaceTokenizer:
    """Stands in for PTBTokenizer, which needs Java."""

    def tokenize(self, captions: Dict[int, List[Dict[str, str]]]) -> Dict[int, List[str]]:
        return {key: [' '.join(c['caption'].lower().split()) for c in value] for key, value in captions.items()}


class _NoMeteor:
    """Stands in for METEOR, which needs Java."""

    def compute_score(self, gts: Dict[int, List[str]], res: Dict[int, List[str]]) -> Tuple[float, List[float]]:
        return 0.0, [0.0] * len(res)


def test_caption_metrics_are_scored_as_one_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('pycocotools')
    pytest.importorskip('pycocoevalcap.eval')
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    from evalscope.benchmarks.refcoco import evaluation_lib

    monkeypatch.setattr(evaluation_lib, 'PTBTokenizer', _WhitespaceTokenizer)
    monkeypatch.setattr(evaluation_lib, 'Meteor', _NoMeteor)
    adapter = get_benchmark('refcoco', TaskConfig(datasets=['refcoco']))

    sample_scores = []
    for index, (prediction, references) in enumerate(zip(PREDICTIONS, REFERENCES)):
        state = TaskState(
            model='mock',
            sample=Sample(
                input='q', target=str(references), metadata={'answer': references, 'eval_mode': 'bbox'}, id=index
            ),
            output=ModelOutput.from_content('mock', prediction),
            completed=True,
        )
        sample_scores.append(adapter.calculate_metrics(state))

    tokenizer = _WhitespaceTokenizer()
    gts = tokenizer.tokenize({i: [{'caption': ref} for ref in refs] for i, refs in enumerate(REFERENCES)})
    res = tokenizer.tokenize({i: [{'caption': pred}] for i, pred in enumerate(PREDICTIONS)})
    corpus_cider, _ = Cider().compute_score(gts, res)
    corpus_bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    assert corpus_cider > 0

    aggregates = {
        (agg.metric_name, agg.dimensions.get('ngram')): agg.score for agg in adapter.aggregate_scores(sample_scores)
    }
    assert aggregates[('cider', None)] == pytest.approx(corpus_cider)
    assert aggregates[('bleu', 4)] == pytest.approx(corpus_bleu[3])
    assert adapter._benchmark_meta.evaluation_version == 'v1.1'
