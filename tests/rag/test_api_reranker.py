import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('langchain_openai')
pytest.importorskip('mteb')
pytest.importorskip('sentence_transformers')

from evalscope.backend.rag_eval.models import load_model
from evalscope.backend.rag_eval.models.reranker import APIReranker


class MockResponse:

    def __init__(self, data):
        self.data = data
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_api_reranker_predict(monkeypatch):
    calls = []

    def mock_post(self, url, headers, json, timeout):
        calls.append({'url': url, 'headers': headers, 'json': json, 'timeout': timeout})
        return MockResponse({
            'results': [
                {
                    'index': 1,
                    'relevance_score': 0.2,
                },
                {
                    'index': 0,
                    'relevance_score': 0.9,
                },
            ]
        })

    monkeypatch.setattr('evalscope.backend.rag_eval.models.reranker.requests.Session.post', mock_post)

    model = APIReranker(
        model_name='Qwen3-Reranker-8B',
        api_base='https://aiping.cn/api/v1',
        api_key='test-key',
        batch_size=10,
    )
    scores = model.predict([['query', 'document 0', None], ['query', 'document 1', None]])

    assert torch.equal(scores, torch.tensor([0.9, 0.2]))
    assert calls == [{
        'url': 'https://aiping.cn/api/v1/rerank',
        'headers': {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-key',
        },
        'json': {
            'model': 'Qwen3-Reranker-8B',
            'query': 'query',
            'documents': ['document 0', 'document 1'],
            'top_n': 2,
            'return_documents': False,
        },
        'timeout': 60,
    }]


def test_api_reranker_keeps_explicit_endpoint(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    def mock_post(self, url, headers, json, timeout):
        return MockResponse({'results': [{'index': 0, 'score': 0.8}]})

    monkeypatch.setattr('evalscope.backend.rag_eval.models.reranker.requests.Session.post', mock_post)

    model = APIReranker(
        model_name='reranker',
        api_base='https://example.com/v1/reranks',
        batch_size=1,
    )

    assert model.rerank_url == 'https://example.com/v1/reranks'
    assert torch.equal(model.predict([['query', 'document']]), torch.tensor([0.8]))


def test_api_reranker_applies_instruction(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    calls = []

    def mock_post(self, url, headers, json, timeout):
        calls.append(json)
        return MockResponse({'results': [{'index': 0, 'score': 0.8}]})

    monkeypatch.setattr('evalscope.backend.rag_eval.models.reranker.requests.Session.post', mock_post)

    model = APIReranker(
        model_name='reranker',
        api_base='https://example.com/v1',
        batch_size=1,
    )

    scores = model.predict([['query', 'document', 'instruction']])

    assert torch.equal(scores, torch.tensor([0.8]))
    assert calls[0]['query'] == 'query instruction'


def test_api_reranker_defaults_none_index(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    def mock_post(self, url, headers, json, timeout):
        return MockResponse({'results': [{'index': None, 'score': 0.8}]})

    monkeypatch.setattr('evalscope.backend.rag_eval.models.reranker.requests.Session.post', mock_post)

    model = APIReranker(
        model_name='reranker',
        api_base='https://example.com/v1',
        batch_size=1,
    )

    assert torch.equal(model.predict([['query', 'document']]), torch.tensor([0.8]))


def test_api_reranker_uses_openai_api_key_env(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key')

    model = APIReranker(
        model_name='reranker',
        api_base='https://example.com/v1',
        batch_size=1,
    )

    assert model.headers['Authorization'] == 'Bearer env-key'


def test_load_api_cross_encoder():
    model = load_model({
        'model_name_or_path': 'Qwen3-Reranker-8B',
        'model_name': 'Qwen3-Reranker-8B',
        'api_base': 'https://aiping.cn/api/v1',
        'api_key': 'test-key',
        'is_cross_encoder': True,
    })

    assert isinstance(model, APIReranker)


def test_load_api_cross_encoder_passes_max_seq_length():
    model = load_model({
        'model_name': 'reranker',
        'api_base': 'https://example.com/v1',
        'api_key': 'test',
        'is_cross_encoder': True,
        'max_seq_length': 1024,
    })

    assert isinstance(model, APIReranker)
    assert model.max_seq_length == 1024
    assert model._max_chars == 1024 * 3


def test_api_reranker_truncates_long_texts(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    calls = []

    def mock_post(self, url, headers, json, timeout):
        calls.append(json)
        return MockResponse({'results': [{'index': 0, 'relevance_score': 0.5}]})

    monkeypatch.setattr('evalscope.backend.rag_eval.models.reranker.requests.Session.post', mock_post)

    model = APIReranker(
        model_name='reranker',
        api_base='https://example.com/v1',
        batch_size=10,
        max_seq_length=10,
    )

    long_query = 'q' * 100
    long_doc = 'd' * 100
    model.predict([[long_query, long_doc]])

    assert len(calls[0]['query']) == 10 * 3
    assert len(calls[0]['documents'][0]) == 10 * 3


def test_api_reranker_no_truncation_short_texts(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    calls = []

    def mock_post(self, url, headers, json, timeout):
        calls.append(json)
        return MockResponse({'results': [{'index': 0, 'score': 0.9}]})

    monkeypatch.setattr('evalscope.backend.rag_eval.models.reranker.requests.Session.post', mock_post)

    model = APIReranker(
        model_name='reranker',
        api_base='https://example.com/v1',
        batch_size=10,
        max_seq_length=512,
    )

    model.predict([['short query', 'short doc']])

    assert calls[0]['query'] == 'short query'
    assert calls[0]['documents'] == ['short doc']
