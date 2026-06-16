import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('langchain_openai')
pytest.importorskip('mteb')

import langchain_openai.embeddings

from evalscope.backend.rag_eval.models import load_model
from evalscope.backend.rag_eval.models.encoder import APIEncoder


class MockOpenAIEmbeddings:
    """Mock for langchain_openai.OpenAIEmbeddings that records calls."""

    def __init__(self, *, model, base_url, api_key, dimensions, check_embedding_ctx_length):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.dimensions = dimensions
        self.check_embedding_ctx_length = check_embedding_ctx_length

    def embed_documents(self, texts, chunk_size=None, **kwargs):
        self._last_texts = texts
        return [[0.1, 0.2, 0.3]] * len(texts)


def _make_encoder(monkeypatch, **overrides):
    """Create an APIEncoder with mocked OpenAIEmbeddings."""
    monkeypatch.setattr(langchain_openai.embeddings, 'OpenAIEmbeddings', MockOpenAIEmbeddings)

    defaults = dict(
        model_name='test-model',
        api_base='http://localhost:8000/v1',
        api_key='test-key',
        dimensions=1024,
        max_seq_length=10,
        batch_size=100,
    )
    defaults.update(overrides)
    return APIEncoder(**defaults)


def test_encode_basic(monkeypatch):
    encoder = _make_encoder(monkeypatch, max_seq_length=512)
    result = encoder.encode(['hello', 'world'])
    assert result.shape == (2, 3)


def test_encode_truncates_long_texts(monkeypatch):
    encoder = _make_encoder(monkeypatch, max_seq_length=10)
    long_text = 'x' * 100
    short_text = 'hi'
    encoder.encode([long_text, short_text])

    last_texts = encoder._client._last_texts
    assert len(last_texts[0]) == 10 * 3
    assert last_texts[1] == short_text


def test_encode_truncates_after_prompt(monkeypatch):
    encoder = _make_encoder(monkeypatch, max_seq_length=10, prompt='prefix: ')
    long_text = 'y' * 100

    try:
        from mteb.types import PromptType
        prompt_type = PromptType.query
    except ImportError:
        prompt_type = 'query'

    encoder.encode([long_text], prompt_type=prompt_type)

    last_texts = encoder._client._last_texts
    assert last_texts[0].startswith('prefix: ')
    assert len(last_texts[0]) == 10 * 3


def test_encode_no_truncation_when_within_limit(monkeypatch):
    encoder = _make_encoder(monkeypatch, max_seq_length=512)
    text = 'short text'
    encoder.encode([text])
    assert encoder._client._last_texts == [text]


def test_all_init_params_stored(monkeypatch):
    encoder = _make_encoder(
        monkeypatch,
        model_name='my-model',
        api_base='http://example.com/v1',
        api_key='key-123',
        dimensions=768,
        max_seq_length=256,
        batch_size=32,
        prompt='query: ',
        prompts={'task1': 'prompt1'},
    )

    assert encoder.model_name_or_path == 'my-model'
    assert encoder.max_seq_length == 256
    assert encoder._max_chars == 256 * 3
    assert encoder.batch_size == 32
    assert encoder.prompt == 'query: '
    assert encoder.prompts == {'task1': 'prompt1'}
    assert encoder._client.model == 'my-model'
    assert encoder._client.base_url == 'http://example.com/v1'
    assert encoder._client.api_key == 'key-123'
    assert encoder._client.dimensions == 768


def test_load_model_creates_api_encoder(monkeypatch):
    monkeypatch.setattr(langchain_openai.embeddings, 'OpenAIEmbeddings', MockOpenAIEmbeddings)

    model = load_model({
        'model_name': 'embed-model',
        'api_base': 'http://localhost/v1',
        'api_key': 'test',
        'dimensions': 512,
        'max_seq_length': 1024,
        'encode_kwargs': {'batch_size': 64},
    })

    assert isinstance(model, APIEncoder)
    assert model.max_seq_length == 1024
    assert model._max_chars == 1024 * 3
    assert model.batch_size == 64


def test_batch_size_from_encode_kwargs(monkeypatch):
    encoder = _make_encoder(monkeypatch, batch_size=2)
    texts = ['a', 'b', 'c', 'd', 'e']
    encoder.encode(texts)
    assert encoder._client._last_texts is not None
