import pytest

from evalscope.benchmarks.bfcl.v4.bfcl_v4_adapter import _normalize_openai_base_url


@pytest.mark.parametrize(
    ('api_url', 'expected'),
    [
        ('https://example.test/v1', 'https://example.test/v1'),
        ('https://example.test/v1/', 'https://example.test/v1'),
        ('https://example.test/v1/chat/completions', 'https://example.test/v1'),
        ('https://example.test/v1/chat/completions/', 'https://example.test/v1'),
        ('  https://example.test/v1/chat/completions/  ', 'https://example.test/v1'),
        ('', ''),
        ('   ', ''),
        (None, ''),
    ],
)
def test_normalize_openai_base_url(api_url, expected):
    assert _normalize_openai_base_url(api_url) == expected
