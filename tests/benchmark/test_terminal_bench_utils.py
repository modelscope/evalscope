import asyncio
from typing import Any

import pytest

pytest.importorskip('harbor')

from evalscope.api.model import ModelOutput, ModelUsage
from evalscope.benchmarks.terminal_bench.utils import HarborLLM


class _ModelDouble:

    def __init__(self, output: ModelOutput) -> None:
        self._output = output

    def generate(self, input: Any) -> ModelOutput:
        return self._output


def test_harbor_llm_tolerates_missing_usage() -> None:
    output = ModelOutput.from_content(model='mock', content='done')

    response = asyncio.run(HarborLLM(_ModelDouble(output)).call('prompt'))

    assert response.content == 'done'
    assert response.usage is None


def test_harbor_llm_maps_usage() -> None:
    output = ModelOutput.from_content(model='mock', content='done')
    output.usage = ModelUsage(input_tokens=10, output_tokens=4, input_tokens_cache_read=3)

    response = asyncio.run(HarborLLM(_ModelDouble(output)).call('prompt'))

    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 4
    assert response.usage.cache_tokens == 3
