"""Offline decision contracts, metrics, official aggregation and native-pipeline regressions."""

import json
from pathlib import Path

import httpx
import pytest

from evalscope import TaskConfig, run_task
from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.api.metric import SampleScore
from evalscope.api.model import GenerateConfig, Model
from evalscope.api.model.decision import DecisionAnswer, DecisionInput, parse_decision_response
from evalscope.api.model.model import ModelCache
from evalscope.api.registry import get_benchmark
from evalscope.metrics.decision import DecisionAggregator, decision_score
from evalscope.models.systemone import AutoTrustAPI


def question(kind: str, levels: int = 4) -> DecisionInput:
    return DecisionInput(kind=kind, state='Visible facts only.', instructions='Decide using the state.',
                         criteria={'A': 'First', 'B': 'Second'} if kind == 'choice' else
                         [str(i) for i in range(levels)] if kind == 'score' else None)


def answer(request: DecisionInput, index: int = 0) -> dict:
    p = {label: float(i == index) for i, label in enumerate(request.labels)}
    result = {'type': request.kind, 'probabilities': p}
    if request.kind == 'choice':
        result['choice'] = request.labels[index]
    elif request.kind == 'noul':
        result['noul'] = p['yes']
    else:
        result.update(score=float(index), legend=dict(zip(request.labels, request.criteria)))
    return result


@pytest.mark.parametrize('kind', ['choice', 'noul', 'score'])
def test_systemone_contract(kind: str) -> None:
    request = question(kind)
    raw = {'model': 'resolved-v1', 'answers': {'answer': answer(request)}}
    parsed = parse_decision_response(raw, request)
    assert parsed.type == kind
    assert list(parsed.probabilities) == request.labels
    if kind == 'noul':
        del raw['answers']['answer']['probabilities']
        assert parse_decision_response(raw, request).probabilities == {'no': 1, 'yes': 0}


@pytest.mark.parametrize('kind', ['choice', 'noul', 'score'])
def test_autotrust_transport_and_labels(kind: str) -> None:
    request = question(kind, levels=6)
    from evalscope.api.messages import ChatMessageUser

    provider = AutoTrustAPI('jev', 'http://example.test/v1')
    def handle(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert str(req.url) == 'http://example.test/v1/decide'
        assert body['thinking'] == 'off'
        options = body.get('options', ['false', 'true'] if kind == 'noul' else [str(i) for i in range(6)])
        return httpx.Response(200, json={'model': 'jev-v1', 'kind': kind, 'options': options,
                                         'probabilities': [1.0] + [0.0] * (len(options) - 1),
                                         'choice_index': 0, 'choice': options[0],
                                         'usage': {'prompt_tokens': 20, 'completion_tokens': 1}})
    provider.client.close()
    provider.client = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        output = Model(provider, GenerateConfig(retries=1)).generate([
            ChatMessageUser(content='Facts', internal={'systemone': request.model_dump()})])
        assert output.usage.total_tokens == 21
        assert list(output.metadata['decision']['probabilities']) == request.labels
    finally:
        provider.client.close()


@pytest.mark.parametrize('levels', [2, 3, 4, 5, 7, 10])
def test_no_silent_score_remapping(levels: int) -> None:
    with pytest.raises(ValueError, match='six Score levels'):
        question('score', levels).autotrust_payload()


@pytest.mark.parametrize('p', [float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_binary_probability(p: float) -> None:
    with pytest.raises(ValueError):
        parse_decision_response({'model': 'v1', 'answers': {'answer': {'type': 'noul', 'noul': p}}}, question('noul'))


def test_hand_computed_probability_metrics() -> None:
    a = DecisionAnswer(type='choice', choice='B', probabilities={'A': 0.25, 'B': 0.75})
    s = decision_score(a, 'B', ['A', 'B'])
    assert s.value['acc'] == 1
    assert s.value['nll'] == pytest.approx(-__import__('math').log(0.75))
    assert s.value['brier'] == pytest.approx(0.125)
    binary = DecisionAnswer(type='noul', noul=0.75, probabilities={'no': 0.25, 'yes': 0.75})
    assert decision_score(binary, 'yes', ['no', 'yes']).value['brier'] == pytest.approx(0.0625)
    zero = DecisionAnswer(choice='A', probabilities={'A': 1, 'B': 0})
    assert decision_score(zero, 'B', ['A', 'B']).value['nll'] > 20


def test_continuous_gold_is_not_discretized() -> None:
    request = question('score', levels=6)
    a = DecisionAnswer.model_validate(answer(request, 3))
    s = decision_score(a, 2.7, request.labels, continuous=True)
    assert s.value == {'mae': pytest.approx(0.3)}
    assert s.metadata['decision']['gold'] == 2.7


def test_cross_sample_calibration_and_auc() -> None:
    samples = []
    for index, (p, gold) in enumerate([(0.1, 'no'), (0.4, 'yes'), (0.8, 'yes'), (0.9, 'no')]):
        a = DecisionAnswer(type='noul', noul=p, probabilities={'no': 1 - p, 'yes': p})
        samples.append(SampleScore(score=decision_score(a, gold, ['no', 'yes']), sample_id=index))
    scores = {s.metric_name: s.score for s in DecisionAggregator()(samples)}
    assert scores['auroc'] == pytest.approx(0.5)
    assert scores['auprc'] == pytest.approx(7 / 12)
    assert scores['ece'] == pytest.approx(0.45)


def test_official_open_decisions_perfect_and_missing() -> None:
    from evalscope.benchmarks.open_decisions.utils import score

    tasks, predictions = [], []
    for kind in ['choice', 'noul', 'score']:
        for tier in ['easy', 'standard', 'judge', 'hard']:
            request = question(kind)
            task = {'id': f'{kind}-{tier}', 'question': {'type': kind}, 'labels': request.labels,
                    'expected': request.labels[-1], 'tier': tier}
            tasks.append(task)
            predictions.append({'id': task['id'], 'probs': {label: float(label == task['expected']) for label in request.labels}})
    result = score(tasks, predictions)
    assert result['capability'] == pytest.approx(100)
    assert result['items'] == result['predicted'] == 12
    missing = score(tasks, [])
    assert missing['intelligence'] == 0
    assert missing['calibration'] is None
    assert missing['predicted'] == 0


@pytest.mark.parametrize('name,kind', [('jevbench', 'score'), ('open_decisions', 'noul'),
                                      ('general_decision', 'choice'), ('stsb', 'score')])
def test_typed_dataset_native_pipeline(name: str, kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter

    monkeypatch.setattr(ModelCache, '_models', {})
    request = question(kind, 6 if name == 'stsb' else 4)
    record = {'id': 'source-1', 'state': request.state,
              'question': request.systemone_payload('test')['questions']['answer'],
              'expected': request.labels[0], 'family': 'unit', 'tier': 'hard', 'rationale': 'SECRET_GOLD'}
    if name == 'stsb':
        record = {'sentence1': 'Cats.', 'sentence2': 'Dogs.', 'score': 0.1}
    def load(adapter: DefaultDataAdapter) -> tuple:
        sample = adapter.record_to_sample(record)
        sample.id = 0
        return DatasetDict({'default': MemoryDataset([sample])}), None
    monkeypatch.setattr(DefaultDataAdapter, 'load', load)
    calls = []

    def handle(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        payload = json.loads(req.content)
        assert 'SECRET_GOLD' not in json.dumps(payload)
        q = payload['questions']['answer']
        contract = DecisionInput(kind=q['type'], state=payload['state'], instructions=q['instructions'], criteria=q.get('criteria'))
        return httpx.Response(200, json={'model': 'resolved-v1', 'answers': {'answer': answer(contract)},
                                         'usage': {'input_tokens': 10, 'output_tokens': 0}})
    original_init = httpx.Client.__init__
    def mock_init(self: httpx.Client, **kwargs: object) -> None:
        original_init(self, transport=httpx.MockTransport(handle), **kwargs)
    monkeypatch.setattr(httpx.Client, '__init__', mock_init)
    run_task(TaskConfig(model='test', eval_type='systemone_api', api_url='http://example.test/v1',
                        datasets=[name], work_dir=str(tmp_path), no_timestamp=True))
    report = json.loads(next((tmp_path / 'reports').rglob(f'{name}.json')).read_text())
    assert report['execution_summary']['succeeded'] == 1
    assert report['primary_metric_identity'] is not None
    assert (tmp_path / 'reports/report.html').exists()
    run_task(TaskConfig(model='test', eval_type='systemone_api', api_url='http://example.test/v1',
                        datasets=[name], work_dir=str(tmp_path), use_cache=str(tmp_path),
                        rerun_review=True, no_timestamp=True))
    assert len(calls) == 1


def test_pinned_official_reference_fixture() -> None:
    from evalscope.benchmarks.open_decisions.utils import score

    fixture = json.loads((Path(__file__).parent / 'data/open_decisions_official.json').read_text())
    assert fixture['source_revision'] == '5026a13ad6e5a8606810a3a5c4e410d8d6638e71'
    assert score(fixture['tasks'], fixture['predictions']) == fixture['expected']


def test_mixed_kinds_keep_losses_separate_and_one_primary() -> None:
    records = []
    for kind in ['choice', 'noul', 'score']:
        q = question(kind)
        a = DecisionAnswer.model_validate(answer(q))
        records.append(SampleScore(score=decision_score(a, q.labels[0], q.labels, family=kind), sample_id=kind))
    aggregates = DecisionAggregator()(records)
    assert len([a for a in aggregates if a.metric_name == 'accuracy' and a.aggregation == 'mean']) == 1
    assert all(a.dimensions.get('kind') for a in aggregates if a.metric_name == 'brier')
    assert next(a for a in aggregates if a.metric_name == 'family_macro').score == 1


def test_undefined_auc_and_correlations_explain_exclusion() -> None:
    a = DecisionAnswer(type='noul', noul=0.5, probabilities={'yes': 0.5, 'no': 0.5})
    aggregates = DecisionAggregator()([SampleScore(score=decision_score(a, 'yes', ['no', 'yes']), sample_id=0)])
    assert not any(s.metric_name == 'auroc' for s in aggregates)
    assert any('unavailable' in (s.metadata or {}) for s in aggregates)


def test_generated_usage_declares_protocol_and_local_path() -> None:
    from evalscope.utils.doc_utils.generate_dataset_md import extract_adapter_meta
    from evalscope.utils.doc_utils.readme_generator import generate_readme_from_dict

    adapter = get_benchmark('general_decision')
    metadata = extract_adapter_meta(adapter)
    readme = generate_readme_from_dict('general_decision', metadata)
    assert '--eval-type systemone_api' in readme
    assert "eval_type='systemone_api'" in readme
    assert 'default_test.jsonl' in readme
    assert 'local_path' in readme


@pytest.mark.parametrize('provider', ['systemone_api', 'autotrust_api'])
def test_boolq_explicit_noul_and_threshold(provider: str) -> None:
    from evalscope.api.evaluator import TaskState
    from evalscope.api.model import ModelOutput

    config = TaskConfig(model='test', eval_type=provider, datasets=['boolq'],
                        dataset_args={'boolq': {'extra_params': {'decision_kind': 'noul', 'decision_threshold': 0.8}}})
    adapter = get_benchmark('boolq', config)
    sample = adapter.record_to_sample({'passage': 'Paris is in France.', 'question': 'Is Paris in France?', 'answer': True})
    messages = adapter.build_systemone_messages(sample, 'default')
    assert messages[-1].internal['systemone']['kind'] == 'noul'
    a = DecisionAnswer(type='noul', noul=0.7, probabilities={'no': 0.3, 'yes': 0.7})
    state = TaskState(model='test', sample=sample, messages=messages)
    state.output = ModelOutput.from_content('test', 'ANSWER: yes')
    state.output.metadata = {'decision': a.model_dump()}
    result = adapter.match_score('yes', 'yes', 'A', state)
    assert result.value['acc'] == 0
    assert result.metadata['decision']['probabilities'][1] == 0.7


def test_jevbench_public_reference_boundaries() -> None:
    fixture = json.loads((Path(__file__).parent / 'data/jevbench_official.json').read_text())
    for case in fixture['cases']:
        kind, p = case['kind'], case['probabilities']
        fields = {'type': kind, 'probabilities': p}
        if kind == 'choice':
            fields['choice'] = max(p, key=p.get)
        elif kind == 'noul':
            fields['noul'] = p['yes']
        else:
            fields['score'] = sum(int(k) * v for k, v in p.items())
            fields['legend'] = {label: label for label in p}
        a = DecisionAnswer.model_validate(fields)
        got = decision_score(a, case['gold'], case['labels'],
                             normalize=abs(sum(p.values()) - 1) > 0.001, noul_use_argmax=True)
        expected = case['expected']
        assert got.metadata['decision']['predicted'] == expected['predicted']
        assert bool(got.value['acc']) == expected['correct']
        if kind == 'score':
            assert got.metadata['decision']['expected'] == pytest.approx(expected['ordinal_ev'])
        assert got.metadata['probabilities_normalized_for_scoring'] == expected['renormalized']


def test_ordinal_errors_do_not_mix_different_scales() -> None:
    samples = []
    for levels in [4, 6]:
        q = question('score', levels)
        samples.append(SampleScore(score=decision_score(DecisionAnswer.model_validate(answer(q)), '0', q.labels), sample_id=levels))
    aggregates = DecisionAggregator()(samples)
    mae = [s for s in aggregates if s.metric_name == 'mae']
    assert len(mae) == 2
    assert {s.dimensions['levels'] for s in mae} == {4, 6}
    assert all(s.num == 1 for s in mae)


@pytest.mark.parametrize('name', ['jevbench', 'open_decisions'])
def test_all_failed_predictions_keep_official_denominator(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter

    monkeypatch.setattr(ModelCache, '_models', {})
    record = {'id': 'missing-1', 'state': 'Facts.', 'question': {'type': 'noul', 'instructions': 'Is it true?'},
              'expected': 'yes', 'family': 'unit', 'tier': 'hard'}
    def load(adapter: DefaultDataAdapter) -> tuple:
        sample = adapter.record_to_sample(record)
        sample.id = 0
        return DatasetDict({'default': MemoryDataset([sample])}), None
    monkeypatch.setattr(DefaultDataAdapter, 'load', load)
    original_init = httpx.Client.__init__
    def mock_init(self: httpx.Client, **kwargs: object) -> None:
        original_init(self, transport=httpx.MockTransport(lambda req: httpx.Response(200, json={'model': 'v1'})), **kwargs)
    monkeypatch.setattr(httpx.Client, '__init__', mock_init)
    run_task(TaskConfig(model='test', eval_type='systemone_api', api_url='http://example.test/v1', datasets=[name],
                        ignore_errors=True, generation_config={'retries': 1}, work_dir=str(tmp_path), no_timestamp=True))
    report = json.loads(next((tmp_path / 'reports').rglob(f'{name}.json')).read_text())
    assert report['execution_summary']['requested'] == report['execution_summary']['errored'] == 1
    assert report['execution_summary']['succeeded'] == 0
    serialized = json.dumps(report)
    assert 'missing_predictions_count_as_wrong' in serialized
    if name == 'jevbench':
        assert report['primary_metric_identity']['name'] == 'public_accuracy'
    else:
        assert report['primary_metric_identity'] is None


@pytest.mark.parametrize('usage', [[], False, '', {'input_tokens': -1}, {'prompt_tokens': 1.5}, {'output_tokens': True}])
def test_malformed_usage_is_not_a_success(usage: object) -> None:
    from evalscope.api.model.decision import parse_decision_usage

    with pytest.raises(ValueError):
        parse_decision_usage({'usage': usage})


def test_public_denominator_stays_within_each_selected_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter

    monkeypatch.setattr(ModelCache, '_models', {})
    def load(adapter: DefaultDataAdapter) -> tuple:
        datasets = {}
        for index, subset in enumerate(['original', 'hard']):
            sample = adapter.record_to_sample({'id': subset, 'state': 'Facts.', 'family': 'unit',
                'question': {'type': 'choice', 'instructions': 'Choose.', 'criteria': {'A': 'First', 'B': 'Second'}},
                'expected': 'A'})
            sample.id = index
            datasets[subset] = MemoryDataset([sample])
        return DatasetDict(datasets), None
    monkeypatch.setattr(DefaultDataAdapter, 'load', load)
    original_init = httpx.Client.__init__
    def mock_init(self: httpx.Client, **kwargs: object) -> None:
        original_init(self, transport=httpx.MockTransport(lambda req: httpx.Response(200, json={
            'model': 'v1', 'answers': {'answer': {'type': 'choice', 'choice': 'A', 'probabilities': {'A': 1, 'B': 0}}}})), **kwargs)
    monkeypatch.setattr(httpx.Client, '__init__', mock_init)
    run_task(TaskConfig(model='test', eval_type='systemone_api', api_url='http://example.test/v1', datasets=['jevbench'],
                        dataset_args={'jevbench': {'subset_list': ['original', 'hard']}}, work_dir=str(tmp_path), no_timestamp=True))
    report = json.loads(next((tmp_path / 'reports').rglob('jevbench.json')).read_text())
    primary = next(m for m in report['metrics'] if m['identity']['name'] == 'public_accuracy')
    assert primary['num'] == 2
    assert primary['score'] == 1
    assert [subset['num'] for category in primary['categories'] for subset in category['subsets']] == [1, 1]


def test_continuous_score_risk_uses_normalized_mae() -> None:
    q = question('score', levels=6)
    score = decision_score(DecisionAnswer.model_validate(answer(q, 3)), 2.7, q.labels, continuous=True)
    aggregates = DecisionAggregator()([SampleScore(score=score, sample_id=0)])
    curve = next(s for s in aggregates if s.metric_name == 'aurc')
    assert curve.dimensions['risk'] == 'normalized_mae'
    assert curve.score == pytest.approx(0.3 / 5)


def test_average_precision_is_defined_for_all_positive_gold() -> None:
    a = DecisionAnswer(type='noul', noul=0.5, probabilities={'yes': 0.5, 'no': 0.5})
    aggregates = DecisionAggregator()([SampleScore(score=decision_score(a, 'yes', ['no', 'yes']), sample_id=0)])
    assert next(s.score for s in aggregates if s.metric_name == 'auprc') == 1
    assert not any(s.metric_name == 'auroc' for s in aggregates)


def test_open_decisions_rejects_rounding_outside_official_tolerance() -> None:
    from evalscope.api.evaluator import TaskState
    from evalscope.api.metric import MetricUnavailableError
    from evalscope.api.model import ModelOutput

    adapter = get_benchmark('open_decisions', TaskConfig(model='test', eval_type='systemone_api', datasets=['open_decisions']))
    q = question('score')
    sample = adapter.record_to_sample({'id': 'rounded', 'state': q.state,
        'question': q.systemone_payload('test')['questions']['answer'], 'expected': '1', 'tier': 'hard'})
    a = DecisionAnswer(type='score', score=1.51, legend=dict(zip(q.labels, q.criteria)),
                       probabilities={'0': 0.26, '1': 0.26, '2': 0.25, '3': 0.25})
    output = ModelOutput.from_content('test', '1.51')
    output.metadata = {'decision': a.model_dump()}
    with pytest.raises(MetricUnavailableError, match='sum tolerance'):
        adapter.match_score('1.51', '1.51', '1', TaskState(model='test', sample=sample, output=output))
