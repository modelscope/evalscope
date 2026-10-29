"""CrossNER prompts and scores must use the entity labels of each sample's own domain subset."""
from typing import Dict, List

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

# Label names as published in extraordinarylab/cross-ner.
RECORDS = {
    'ai': {
        'tokens': ['BERT', 'beats', 'GPT', 'in', 'accuracy', 'at', 'Google'],
        'ner_tags': ['B-ALGORITHM', 'O', 'B-ALGORITHM', 'O', 'B-METRIC', 'O', 'B-ORGANIZATION'],
    },
    'science': {
        'tokens': ['Mars', 'orbits', 'the', 'Sun'],
        'ner_tags': ['B-ASTRONOMICAL_OBJECT', 'O', 'O', 'B-ASTRONOMICAL_OBJECT'],
    },
}


def _load_like_the_pipeline() -> tuple:
    """Convert each subset's record while that subset loads, then hand back control as the pipeline does."""
    pytest.importorskip('seqeval')
    adapter = get_benchmark('cross_ner', TaskConfig(datasets=['cross_ner']))
    samples: Dict[str, Sample] = {}
    for subset, record in RECORDS.items():
        with adapter._temporary_attribute('current_subset_name', subset):
            samples[subset] = adapter.record_to_sample(record)
    return adapter, samples


def test_prompt_lists_the_sample_subset_entity_tags() -> None:
    adapter, samples = _load_like_the_pipeline()

    prompt = adapter.format_prompt_template(samples['ai'])

    assert '<algorithm>' in prompt and '<metrics>' in prompt
    assert '<astronomical_object>' not in prompt


@pytest.mark.parametrize('subset', list(RECORDS))
def test_gold_annotation_scores_perfectly_in_every_subset(subset: str) -> None:
    adapter, samples = _load_like_the_pipeline()
    sample = samples[subset]
    state = TaskState(model='scripted-model', sample=sample)

    score = adapter.match_score(sample.target, sample.target, sample.target, state)

    tags: List[str] = RECORDS[subset]['ner_tags']
    assert score.metadata['y_pred'] == tags
    assert score.value == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0}
