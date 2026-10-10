"""NER annotations must retain each entity's occurrence and BIO boundary."""

import json
from pathlib import Path
from typing import List

import pytest

from evalscope import run_task
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
from evalscope.config import TaskConfig
from evalscope.models.mockllm import MockLLM
from evalscope.utils.ner import clean_prediction, create_target_text, xml_to_bio_tags

ENTITY_MAP = {'PER': 'person', 'LOC': 'location', 'ORG': 'organization'}
REVERSE_MAP = {value: key for key, value in ENTITY_MAP.items()}

CASES = [
    (['John', 'Alice'], ['B-PER', 'B-PER']),
    (['John', 'met', 'John'], ['B-PER', 'O', 'B-PER']),
    (['Paris', 'and', 'Paris'], ['O', 'O', 'B-LOC']),
    (['John', 'Smith', 'Alice', 'Jones'], ['B-PER', 'I-PER', 'B-PER', 'I-PER']),
    (['王', '小明', '李', '梅'], ['B-PER', 'I-PER', 'B-PER', 'I-PER']),
    (['王', '小明', '访问', 'New', 'York'], ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC']),
    (['John', 'visits', 'Paris', 'with', 'Acme'], ['B-PER', 'O', 'B-LOC', 'O', 'B-ORG']),
    (['Paris', 'visits', 'Paris'], ['B-PER', 'O', 'B-LOC']),
    (['John', 'Smith', 'meets', 'John', 'Smith'], ['O', 'O', 'O', 'B-PER', 'I-PER']),
    (['John', 'visits', 'Paris'], ['O', 'O', 'O']),
]

NER_BENCHMARKS = [
    'anat_em', 'bc2gm', 'bc4chemd', 'bc5cdr', 'broad_twitter_corpus', 'conll2003', 'conllpp', 'copious',
    'cross_ner', 'fin_ner', 'genia_ner', 'harvey_ner', 'jnlpba', 'jnlpba_rare', 'mit_movie_trivia',
    'mit_restaurant', 'multi_nerd', 'ncbi', 'ontonotes5', 'tweebank_ner', 'tweet_ner_7', 'wnut2017',
]


@pytest.mark.parametrize(('tokens', 'tags'), CASES)
def test_target_annotation_round_trips_to_bio(tokens: List[str], tags: List[str]) -> None:
    prediction = create_target_text(tokens, tags, ENTITY_MAP)

    assert xml_to_bio_tags(clean_prediction(prediction), tokens, REVERSE_MAP) == tags


@pytest.mark.parametrize(('tokens', 'tags'), CASES[:-1])
def test_perfect_annotation_has_perfect_seqeval_scores(tokens: List[str], tags: List[str]) -> None:
    pytest.importorskip('seqeval')
    adapter = get_benchmark('conll2003', TaskConfig(datasets=['conll2003']))
    sample = adapter.record_to_sample({'tokens': tokens, 'ner_tags': tags})
    state = TaskState(model='scripted-model', sample=sample)
    prediction = sample.target

    score = adapter.match_score(prediction, prediction, state.target, state)

    assert score.metadata['y_true'] == tags
    assert score.metadata['y_pred'] == tags
    assert score.value == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0}
    aggregated = adapter.aggregate_scores([SampleScore(score=score, sample_id=0)])
    assert {item.metric_name: item.score for item in aggregated} == score.value


def test_annotation_inside_token_does_not_mark_touching_token() -> None:
    assert xml_to_bio_tags('<person>John</person>son works', ['Johnson', 'works'], REVERSE_MAP) == [
        'B-PER', 'O'
    ]


def test_modified_prediction_keeps_existing_alignment_fallback() -> None:
    assert xml_to_bio_tags('Answer: <person>John</person> works', ['John', 'works'], REVERSE_MAP) == [
        'B-PER', 'O'
    ]


def test_nested_markup_keeps_existing_alignment_fallback() -> None:
    prediction = clean_prediction('<person><b>John</b></person> works')
    assert xml_to_bio_tags(prediction, ['John', 'works'], REVERSE_MAP) == ['O', 'O']


@pytest.mark.parametrize('benchmark', NER_BENCHMARKS)
def test_ner_benchmark_scores_preserve_entity_boundaries(benchmark: str) -> None:
    pytest.importorskip('seqeval')
    adapter = get_benchmark(benchmark, TaskConfig(datasets=[benchmark]))
    adapter.current_subset_name = adapter.benchmark_meta.subset_list[0]
    adapter.setup_entity_mappings()
    entity_type = next(iter(adapter.entity_type_map))
    tags = [f'B-{entity_type}', f'B-{entity_type}', 'O', f'B-{entity_type}']
    sample = adapter.record_to_sample({'tokens': ['Delta', 'Echo', 'met', 'Delta'], 'ner_tags': tags})
    state = TaskState(model='scripted-model', sample=sample)

    score = adapter.match_score(sample.target, sample.target, state.target, state)

    assert adapter.benchmark_meta.evaluation_version == 'v1.1'
    assert score.metadata['y_pred'] == tags
    assert score.value == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0}


def test_ner_reviews_require_rescoring_after_span_alignment_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip('seqeval')
    tokens, tags = CASES[1]
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'test.jsonl').write_text(json.dumps({'tokens': tokens, 'ner_tags': tags}) + '\n')
    work = tmp_path / 'run'
    config = dict(
        model='offline', eval_type='mock_llm', datasets=['conll2003'], no_timestamp=True,
        work_dir=str(work), judge={'strategy': 'rule'},
        dataset_args={'conll2003': {'local_path': str(data), 'few_shot_num': 0, 'subset_list': ['default']}},
    )
    metadata = get_benchmark('conll2003').benchmark_meta
    assert metadata.evaluation_version == 'v1.1'
    metadata = BENCHMARK_REGISTRY['conll2003']
    monkeypatch.setattr(MockLLM, 'default_output', create_target_text(tokens, tags, ENTITY_MAP))
    monkeypatch.setattr(metadata, 'evaluation_version', 'v1.0')
    run_task(TaskConfig(**config))
    prediction_path = work / 'predictions/offline/conll2003_default.jsonl'
    previous_predictions = prediction_path.read_bytes()
    monkeypatch.setattr(metadata, 'evaluation_version', 'v1.1')
    with pytest.raises(ValueError, match='rerun_review=True'):
        run_task(TaskConfig(**config, use_cache=str(work)))
    monkeypatch.setattr(MockLLM, 'generate', lambda *args, **kwargs: pytest.fail('Must reuse saved predictions'))

    run_task(TaskConfig(**config, use_cache=str(work), rerun_review=True))

    assert prediction_path.read_bytes() == previous_predictions
    review = json.loads((work / 'reviews/offline/conll2003_default.jsonl').read_text())
    assert review['sample_score']['score']['metadata']['y_pred'] == tags
    assert review['sample_score']['score']['value']['f1'] == 1.0
    report = json.loads((work / 'reports/offline/conll2003.json').read_text())
    assert {metric['identity']['name']: metric['score'] for metric in report['metrics']} == {
        'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0,
    }
