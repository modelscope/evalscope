"""Offline contract, transport and native-pipeline tests for System One Choice."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evalscope import TaskConfig, run_task
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.model import ChoiceQuestion, ChoiceRequest, ChoiceResult, GenerateConfig, ModelOutput
from evalscope.api.model.model import ModelCache, get_model_with_task_config
from evalscope.api.registry import get_benchmark
from evalscope.evaluation_versioning import ResolvedBenchmarkSpec, build_benchmark_identity
from evalscope.models.systemone import SystemOneAPI
from evalscope.utils.multi_choices import answer_character


def request(options: int = 2) -> ChoiceRequest:
    return ChoiceRequest(
        state={'question': 'Choose the best answer.'},
        question=ChoiceQuestion(instructions='Choose one.', criteria={answer_character(i): f'Option {i}' for i in range(options)}),
    )


def response(payload: dict, choice: str | None = None) -> dict:
    criteria = payload['questions']['answer']['criteria']
    choice = choice or next(iter(criteria))
    return {
        'model': 'resolved-model-v1',
        'request_id': 'request-1',
        'answers': {'answer': {'type': 'choice', 'choice': choice, 'confidence': 0.99,
                               'probabilities': {key: float(key == choice) for key in criteria}}},
        'usage': {'input_tokens': 12},
        'latency_ms': 42,
    }


@pytest.mark.parametrize('credentials,expected_key', [
    ({}, 'typesafe-fixture'),
    ({'api_key': None}, 'typesafe-fixture'),
    ({'api_key': ''}, 'typesafe-fixture'),
    ({'api_key': 'EMPTY'}, 'typesafe-fixture'),
    ({'api_key': 'explicit-fixture'}, 'explicit-fixture'),
])
def test_typesafe_authentication_from_task_config(
    credentials: dict, expected_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TYPESAFE_API_KEY', 'typesafe-fixture')
    monkeypatch.setattr(ModelCache, '_models', {})
    seen = []

    def handle(req: httpx.Request) -> httpx.Response:
        seen.append(req.headers.get('Authorization'))
        assert str(req.url) == 'https://api.typesafe.ai/v1/systemone'
        return httpx.Response(200, json=response(json.loads(req.content)))

    original = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    config = TaskConfig(model='jev-1.13.0', eval_type='systemone_api', api_url='https://api.typesafe.ai/v1',
                        **credentials)
    model = get_model_with_task_config(config)
    try:
        output = model.generate_choice(request(), GenerateConfig(retries=1))
        assert seen == [f'Bearer {expected_key}']
        assert model.api.api_key == expected_key
        assert expected_key not in output.model_dump_json()
    finally:
        model.api.client.close()


@pytest.mark.parametrize('env_key', [None, '', 'EMPTY'])
def test_missing_typesafe_key_does_not_send_placeholder(env_key: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    if env_key is None:
        monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    else:
        monkeypatch.setenv('TYPESAFE_API_KEY', env_key)
    provider = SystemOneAPI('test', 'https://api.typesafe.ai/v1', api_key='EMPTY')
    try:
        assert 'Authorization' not in provider.client.headers
        assert provider.api_key is None
    finally:
        provider.client.close()


@pytest.mark.parametrize('base_url', [
    'https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
    'https://example.test/v1',
    'https://api.typesafe.ai.example.test/v1',
    'http://api.typesafe.ai/v1',
])
def test_typesafe_environment_key_is_scoped_to_official_endpoint(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TYPESAFE_API_KEY', 'typesafe-fixture')
    provider = SystemOneAPI('test', base_url, api_key='EMPTY')
    try:
        assert 'Authorization' not in provider.client.headers
        assert provider.api_key is None
    finally:
        provider.client.close()


@pytest.mark.parametrize('options', [2, 4, 10, 77, 255])
def test_wire_request_and_roundtrip(options: int) -> None:
    provider = SystemOneAPI('test', 'https://example.test/compatible-mode/v1')
    seen = []
    def handle(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        seen.append(payload)
        return httpx.Response(200, json=response(payload, choice=list(payload['questions']['answer']['criteria'])[-1]))
    provider.client.close()
    provider.client = httpx.Client(transport=httpx.MockTransport(handle))
    output = provider.generate_choice(request(options), GenerateConfig(retries=1))
    assert output.model == 'resolved-model-v1'
    assert len(output.choice_result.probabilities) == options
    assert output.completion == answer_character(options - 1)
    assert output.metadata['choice_response']['latency_ms'] == 42
    assert output.metadata['choice_request'] == seen[0]
    assert output.usage.total_tokens == 12
    assert output.choices[0].logprobs is None
    assert ModelOutput.model_validate_json(output.model_dump_json()).choice_result == output.choice_result
    provider.client.close()


@pytest.mark.parametrize('changes', [
    {'choice': 'unknown'}, {'type': 'noul'}, {'probabilities': {'A': 0.2, 'B': 0.2}},
    {'probabilities': {'A': float('nan'), 'B': 0}}, {'probabilities': {'A': -0.1, 'B': 1.1}},
    {'choice': 'B'}, {'confidence': float('nan')},
])
def test_invalid_choice_result(changes: dict) -> None:
    answer = {'type': 'choice', 'choice': 'A', 'probabilities': {'A': 1, 'B': 0}, 'confidence': 0.99}
    with pytest.raises(ValueError):
        ChoiceResult.model_validate({**answer, **changes})


@pytest.mark.parametrize('status,attempts', [(400, 1), (401, 1), (422, 1), (429, 2), (529, 2), (503, 2)])
def test_transport_retry_policy(status: int, attempts: int) -> None:
    provider = SystemOneAPI('test', 'https://example.test/v1')
    seen = []
    def handle(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if len(seen) == 1:
            return httpx.Response(status, json={'error': 'test failure'})
        return httpx.Response(200, json=response(json.loads(req.content)))
    provider.client.close()
    provider.client = httpx.Client(transport=httpx.MockTransport(handle))
    if attempts == 1:
        with pytest.raises(httpx.HTTPStatusError):
            provider.generate_choice(request(), GenerateConfig(retries=2, retry_interval=0))
    else:
        provider.generate_choice(request(), GenerateConfig(retries=2, retry_interval=0))
    assert len(seen) == attempts
    provider.client.close()


def test_timeout_retry_and_invalid_response_not_retried() -> None:
    provider = SystemOneAPI('test', 'https://example.test/v1')
    calls = []
    def handle(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if len(calls) == 1:
            raise httpx.ReadTimeout('test timeout', request=req)
        return httpx.Response(200, json={'model': 'test', 'answers': {}})
    provider.client.close()
    provider.client = httpx.Client(transport=httpx.MockTransport(handle))
    with pytest.raises(ValueError, match='exactly'):
        provider.generate_choice(request(), GenerateConfig(retries=3, retry_interval=0))
    assert len(calls) == 2
    provider.client.close()


@pytest.mark.parametrize('config', [{'temperature': 0}, {'max_tokens': 20}, {'stream': True}, {'extra_body': {'think': True}}])
def test_generation_controls_rejected(config: dict) -> None:
    with pytest.raises(ValueError, match='generation_config'):
        SystemOneAPI('test', 'https://example.test/v1', config=GenerateConfig(**config))


AUDITED = ['mmlu', 'ceval', 'cmmlu', 'mmlu_pro', 'arc', 'gpqa_diamond', 'hellaswag', 'winogrande',
           'general_mcq', 'musr', 'super_gpqa']


@pytest.mark.parametrize('name', AUDITED)
def test_audited_adapter_preserves_context_without_gold(name: str) -> None:
    config = TaskConfig(model='test', eval_type='systemone_api', dataset_args={name: {'few_shot_num': 0}})
    adapter = get_benchmark(name, config)
    adapter.validate_choice_config()
    sample = Sample(input='Raw question', choices=['one', 'two'], target='A',
                    metadata={'subject': 'computer_network', 'explanation': 'SECRET_GOLD', 'correct_answer': 'SECRET_GOLD'})
    built = adapter.build_choice_request(sample, 'default')
    assert built.state['question'] == 'Raw question'
    assert built.question.criteria == {'A': 'one', 'B': 'two'}
    assert 'SECRET_GOLD' not in built.model_dump_json()


@pytest.mark.parametrize('name', ['gpqa_diamond', 'super_gpqa'])
def test_fixed_examples(name: str) -> None:
    config = TaskConfig(model='test', eval_type='systemone_api', dataset_args={name: {'few_shot_num': 5}})
    adapter = get_benchmark(name, config)
    built = adapter.build_choice_request(Sample(input='Q', choices=['x', 'y'], target='A'), 'default')
    assert len(built.state['examples'][0]) > 1000


def test_system_prompt_and_selected_demonstrations() -> None:
    config = TaskConfig(model='test', eval_type='systemone_api', dataset_args={'mmlu': {
        'few_shot_num': 1, 'system_prompt': 'Use the supplied evidence.'}})
    adapter = get_benchmark('mmlu', config)
    adapter.fewshot_dataset = DatasetDict({'subject': MemoryDataset([
        Sample(input='DEMO', choices=['x', 'y'], target='B')])})
    built = adapter.build_choice_request(Sample(input='TEST', choices=['x', 'y'], target='A'), 'subject')
    assert built.question.instructions.startswith('Use the supplied evidence.')
    assert 'DEMO' in built.state['examples'][0]
    assert 'TEST' not in built.state['examples'][0]


@pytest.mark.parametrize('overrides', [{'prompt_template': '{question}'}, {'few_shot_prompt_template': '{fewshot}'},
                                      {'filters': {'remove': {}}}, {'extra_params': {'multiple_correct': True}},
                                      {'extra_params': {'use_cot': True}}])
def test_incompatible_adapter_configuration(overrides: dict) -> None:
    config = TaskConfig(model='test', eval_type='systemone_api', dataset_args={'general_mcq': overrides})
    with pytest.raises(ValueError):
        get_benchmark('general_mcq', config).validate_choice_config()


@pytest.mark.parametrize('sample', [Sample(input='Q', choices=['x'], target='A'),
                                   Sample(input='Q', choices=['x', 'y'], target=['A', 'B']),
                                   Sample(input='Q', choices=['x', 'y'], target='C')])
def test_invalid_samples(sample: Sample) -> None:
    adapter = get_benchmark('general_mcq', TaskConfig(model='test', eval_type='systemone_api'))
    with pytest.raises(ValueError):
        adapter.build_choice_request(sample, 'default')


def test_unsupported_benchmark_rejected_before_loading() -> None:
    config = TaskConfig(model='test', eval_type='systemone_api')
    with pytest.raises(ValueError, match='no audited'):
        get_benchmark('gsm8k', config).validate_choice_config()


def test_choice_settings_use_existing_evaluation_identity() -> None:
    config = TaskConfig(model='test', eval_type='systemone_api')
    adapter = get_benchmark('general_mcq', config)
    spec = ResolvedBenchmarkSpec.from_meta(adapter.benchmark_meta, config)
    identity = build_benchmark_identity(spec, 'v1.0', config)
    spec.choice_instructions = 'A different task.'
    assert build_benchmark_identity(spec, 'v1.0', config).fingerprint != identity.fingerprint
    config.eval_type = 'mock_llm'
    generation = ResolvedBenchmarkSpec.from_meta(adapter.benchmark_meta, config)
    assert 'choice_protocol_version' not in generation.fingerprint_dict()


def test_rounded_large_distribution_is_retained() -> None:
    probabilities = {answer_character(i): 0.0 for i in range(77)}
    probabilities.update({'A': 0.70, 'B': 0.20, 'C': 0.06})
    answer = ChoiceResult(choice='A', probabilities=probabilities)
    assert sum(answer.probabilities.values()) == pytest.approx(0.96)
    with pytest.raises(ValueError, match='rounding tolerance'):
        ChoiceResult(choice='A', probabilities={key: 0 for key in probabilities})


def test_anli_round_split_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    class Loader:
        def __init__(self, **kwargs: Any) -> None:
            seen.append((kwargs['subset'], kwargs['split']))

        def load(self) -> MemoryDataset:
            return MemoryDataset([])

    adapter = get_benchmark('anli', TaskConfig(model='test'))
    adapter.load_subset('r3', Loader)
    adapter.load_fewshot_subset('r2', Loader)
    assert seen == [('plain_text', 'test_r3'), ('plain_text', 'train_r2')]
    assert adapter.eval_split == 'test'
    assert adapter.train_split == 'train'


def test_failure_is_excluded_with_execution_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ModelCache, '_models', {})
    data = tmp_path / 'data'
    data.mkdir()
    rows = [{'question': str(i), 'A': 'correct', 'B': 'wrong', 'answer': 'A'} for i in range(2)]
    (data / 'default_val.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))

    def handle(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        if payload['state']['question'] == '1':
            return httpx.Response(200, json={'model': 'test', 'answers': {}})
        return httpx.Response(200, json=response(payload))

    original = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    output = tmp_path / 'outputs'
    run_task(TaskConfig(model='test', eval_type='systemone_api', api_url='https://example.test/v1',
                        datasets=['general_mcq'], dataset_args={'general_mcq': {'local_path': str(data)}},
                        work_dir=str(output), no_timestamp=True, ignore_errors=True))
    report = json.loads(next((output / 'reports').rglob('general_mcq.json')).read_text())
    assert report['execution_summary']['requested'] == 2
    assert report['execution_summary']['succeeded'] == 1
    assert report['execution_summary']['errored'] == 1
    assert report['execution_summary']['incomplete'] is True
    review = json.loads(next((output / 'reviews').rglob('*.jsonl')).read_text())
    assert list(review['sample_score']['score']['value'].values()) == [1.0]


def test_huggingface_documentation_link() -> None:
    from evalscope.utils.doc_utils.readme_generator import _format_dataset_link

    assert 'https://huggingface.co/datasets/google/boolq' in _format_dataset_link('google/boolq', 'huggingface')


NEW_RECORDS = {
    'anli': {'premise': 'Alice is home.', 'hypothesis': 'Alice is home.', 'label': 0, 'reason': 'SECRET_GOLD'},
    'boolq': {'passage': 'Paris is in France.', 'question': 'Is Paris in France?', 'answer': True},
    'banking77': {'text': 'Please activate my card.', 'label': 0, 'label_text': 'activate_my_card'},
    'contract_nli': {'premise': 'No disclosure is allowed.', 'hypothesis': 'Disclosure is allowed.', 'label': 0},
    'reward_bench': {'prompt': 'Say hello.', 'chosen': 'Hello!', 'rejected': 'Goodbye.', 'subset': 'alpacaeval-easy'},
}


@pytest.mark.parametrize('name', list(NEW_RECORDS) + ['ceval'])
def test_native_pipeline_and_reports(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter

    monkeypatch.setattr(ModelCache, '_models', {})
    targets = []
    seen = []
    def load(adapter: DefaultDataAdapter) -> tuple[DatasetDict, None]:
        if name == 'ceval':
            sample = Sample(input='题目', choices=['正确', '错误'], target='A', metadata={'subject': 'computer_network'})
        else:
            sample = adapter.record_to_sample(NEW_RECORDS[name])
        sample.id = 0
        targets.append(sample.target)
        subset = sample.subset_key or adapter.subset_list[0]
        return DatasetDict({subset: MemoryDataset([sample])}), None
    monkeypatch.setattr(DefaultDataAdapter, 'load', load)
    def handle(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        seen.append(payload)
        return httpx.Response(200, json=response(payload, targets[0]))
    original = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    config = TaskConfig(model='test', eval_type='systemone_api', api_url='https://example.test/v1', datasets=[name],
                        dataset_args={name: {'few_shot_num': 0}}, work_dir=str(tmp_path), no_timestamp=True)
    run_task(config)
    prediction = json.loads(next((tmp_path / 'predictions').rglob('*.jsonl')).read_text())
    review = json.loads(next((tmp_path / 'reviews').rglob('*.jsonl')).read_text())
    report = json.loads(next((tmp_path / 'reports').rglob(f'{name}.json')).read_text())
    assert prediction['model_output']['choice_result']['choice'] == targets[0]
    assert prediction['model_output']['metadata']['choice_request'] == seen[0]
    assert 'SECRET_GOLD' not in json.dumps(seen[0])
    assert list(review['sample_score']['score']['value'].values()) == [1.0]
    assert report['execution_summary']['succeeded'] == 1
    assert report['execution_summary']['errored'] == 0
    assert (tmp_path / 'reports' / 'report.html').is_file()
