"""Configured scorers must not share model state or publish unfinished initialization."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any

import pytest

from evalscope.api.metric import SingletonMetric


class ConfiguredMetric(SingletonMetric):
    def _init_once(self, model: str = 'default', *, device: str = 'cpu', **kwargs: Any) -> None:
        self.config = (model, device, kwargs)

    def apply(self, predictions: list[str], references: list[str]) -> list[float]:
        return [1.0] * len(predictions)


@pytest.fixture
def metric_type() -> type[ConfiguredMetric]:
    # A fresh concrete type gives each test a separate cache on both implementations.
    class MetricUnderTest(ConfiguredMetric):
        pass

    return MetricUnderTest


def test_equivalent_constructor_forms_reuse_instance(metric_type: type[ConfiguredMetric]) -> None:
    first = metric_type()
    assert metric_type('default') is first
    assert metric_type(model='default', device='cpu') is first


@pytest.mark.parametrize('kwargs', [{'model': 'other'}, {'device': 'cuda:1'}, {'normalize': False}])
def test_different_configurations_keep_their_state(
    metric_type: type[ConfiguredMetric], kwargs: dict[str, Any]
) -> None:
    original = metric_type()
    changed = metric_type(**kwargs)
    assert changed is not original
    assert original.config == ('default', 'cpu', {})
    assert metric_type(**kwargs) is changed
    assert metric_type() is original


def test_nested_config_values_and_keyword_order(metric_type: type[ConfiguredMetric]) -> None:
    config = {'layers': [1, 2], 'weights': {'x': 0.5, 'y': 1.0}}
    first = metric_type(options=config, flags={'a', 'b'})
    assert metric_type(flags={'b', 'a'}, options={'weights': {'y': 1.0, 'x': 0.5}, 'layers': [1, 2]}) is first
    config['layers'].append(3)
    assert metric_type(options=config, flags={'a', 'b'}) is not first
    assert metric_type(options={'layers': (1, 2), 'weights': {'x': 0.5, 'y': 1.0}}, flags={'a', 'b'}) is not first


@pytest.mark.parametrize(('left', 'right'), [(True, 1), (1, 1.0), ([1], (1,))])
def test_configuration_types_do_not_collide(
    metric_type: type[ConfiguredMetric], left: Any, right: Any
) -> None:
    assert metric_type(option=left) is not metric_type(option=right)


def test_child_class_never_reuses_parent_instance(metric_type: type[ConfiguredMetric]) -> None:
    parent = metric_type()

    class Child(metric_type):
        pass

    child = Child()
    assert type(child) is Child
    assert child is not parent


def test_invalid_arguments_are_not_hidden_by_cache(metric_type: type[ConfiguredMetric]) -> None:
    metric_type()
    with pytest.raises(TypeError):
        metric_type('a', model='b')


def test_unhashable_external_options_bypass_cache(metric_type: type[ConfiguredMetric]) -> None:
    class Option:
        __hash__ = None

    option = Option()
    first = metric_type(option=option)
    second = metric_type(option=option)
    assert first is not second
    assert first.config[2]['option'] is second.config[2]['option'] is option


def test_concurrent_first_use_initializes_once_and_waits() -> None:
    entered = threading.Event()
    duplicate = threading.Event()
    release = threading.Event()
    callers = threading.Barrier(8)
    loads = []

    class SlowMetric(ConfiguredMetric):
        def _init_once(self, model: str = 'default', **kwargs: Any) -> None:
            loads.append(self)
            if len(loads) > 1:
                duplicate.set()
            entered.set()
            assert release.wait(5), 'test did not release the model loader'
            super()._init_once(model, **kwargs)

    def construct(_: int) -> SlowMetric:
        callers.wait(timeout=5)
        instance = SlowMetric()
        assert instance.config == ('default', 'cpu', {})
        return instance

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(construct, i) for i in range(8)]
        try:
            assert entered.wait(5)
            duplicate.wait(0.2)
            assert not any(future.done() for future in futures)
        finally:
            release.set()
        instances = [future.result(timeout=5) for future in futures]
    assert len(loads) == 1
    assert all(instance is instances[0] for instance in instances)


def test_different_configurations_can_initialize_independently() -> None:
    loaders = threading.Barrier(2)

    class ParallelMetric(ConfiguredMetric):
        def _init_once(self, model: str = 'default', **kwargs: Any) -> None:
            loaders.wait(timeout=5)
            super()._init_once(model, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(ParallelMetric, 'first')
        second = pool.submit(ParallelMetric, 'second')
        instances = [first.result(timeout=6), second.result(timeout=6)]
    assert [instance.config[0] for instance in instances] == ['first', 'second']
    assert instances[0] is not instances[1]


def test_failed_initialization_retries_with_a_fresh_instance() -> None:
    attempts = []

    class RetryingMetric(ConfiguredMetric):
        def _init_once(self, **kwargs: Any) -> None:
            attempts.append(self)
            assert not hasattr(self, 'partial'), 'failed instance was reused'
            if len(attempts) == 1:
                self.partial = True
                raise RuntimeError('temporary load failure')
            super()._init_once(**kwargs)

    with pytest.raises(RuntimeError, match='temporary load failure'):
        RetryingMetric()
    good = RetryingMetric()
    assert good is RetryingMetric()
    assert len(attempts) == 2
    assert attempts[0] is not good


def test_waiter_recovers_after_concurrent_initialization_failure() -> None:
    entered = threading.Event()
    release = threading.Event()
    attempts = []

    class RetryingMetric(ConfiguredMetric):
        def _init_once(self) -> None:
            attempts.append(self)
            if len(attempts) == 1:
                entered.set()
                assert release.wait(5)
                self.partial = True
                raise RuntimeError('temporary load failure')
            assert not hasattr(self, 'partial')
            super()._init_once()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(RetryingMetric)
        assert entered.wait(5)
        waiter = pool.submit(RetryingMetric)
        release.set()
        with pytest.raises(RuntimeError, match='temporary load failure'):
            first.result(timeout=5)
        good = waiter.result(timeout=5)
    assert good is RetryingMetric()
    assert len(attempts) == 2
    assert attempts[0] is not good


@pytest.mark.parametrize('order', [('drivel_writing', 'wmt24pp'), ('wmt24pp', 'drivel_writing')])
def test_native_benchmarks_keep_default_bert_models(
    monkeypatch: pytest.MonkeyPatch, order: tuple[str, str]
) -> None:
    from evalscope.api.dataset import Sample
    from evalscope.api.evaluator import TaskState
    from evalscope.api.metric import SampleScore, Score
    from evalscope.api.model import ModelOutput
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig
    from evalscope.metrics.nlp import metrics

    model_scores = {'AI-ModelScope/roberta-large': 0.25, 'AI-ModelScope/xlm-roberta-large': 0.75}
    loads = []

    class Scalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class FakeScorer:
        def __init__(self, model_id_or_path: str, **kwargs: Any) -> None:
            self.model_id = model_id_or_path
            loads.append(model_id_or_path)

        def score(self, predictions: list[str], references: list[str]) -> tuple:
            return None, None, [Scalar(model_scores[self.model_id]) for _ in predictions]

    # Keep the real metric implementation and benchmark batch/aggregation paths.
    class FreshBertScore(metrics.BertScore):
        pass

    scorer_module = ModuleType('evalscope.metrics.nlp.bert_score.scorer')
    scorer_module.BERTScorer = FakeScorer
    monkeypatch.setitem(sys.modules, scorer_module.__name__, scorer_module)
    monkeypatch.setattr(metrics, 'BertScore', FreshBertScore)
    monkeypatch.setattr(metrics, 'check_import', lambda *args, **kwargs: None)
    monkeypatch.setattr('evalscope.benchmarks.wmt.wmt24_adapter.check_import', lambda *args, **kwargs: None)
    for name in (*order, *order):
        adapter = get_benchmark(name, TaskConfig(model='mock', datasets=[name]))
        model_args = adapter.get_metric_args('bert_score')
        # Disable unrelated COMET, BLEU and judge I/O, retaining the real default BERT model.
        adapter = get_benchmark(name, TaskConfig(model='mock', datasets=[name], dataset_args={name: {
            'metric_list': [{'bert_score': model_args}], 'primary_metric': 'bert_score',
        }}))
        state = TaskState(model='mock', sample=Sample(id=0, input='question', target='reference'),
                          output=ModelOutput.from_content('mock', 'candidate'), completed=True)
        scores = adapter.batch_calculate_metrics([state], [SampleScore(sample_id=0, score=Score())])
        restored = SampleScore.model_validate_json(scores[0].model_dump_json())
        aggregates = adapter.aggregate_scores([restored])
        assert restored.score.value['bert_score'] == model_scores[model_args['model_id_or_path']]
        aggregate = next(item for item in aggregates if item.metric_name == 'bert_score')
        assert aggregate.score == model_scores[model_args['model_id_or_path']]
        report = adapter.generate_report({'default': aggregates}, model_name='mock', output_dir='')
        assert report.score == model_scores[model_args['model_id_or_path']]
    assert sorted(loads) == sorted(model_scores)
