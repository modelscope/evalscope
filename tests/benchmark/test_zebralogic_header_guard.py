"""ZebraLogicBench: a malformed header raises a clear ValueError, not a bare assert.

``format_prompt_template`` assumed the first header column is "House". A bare
``assert`` gave no diagnostic and vanished under ``python -O``, so a dataset
with a reordered or renamed header silently built the wrong prompt template.
"""
import json

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def _adapter():
    return get_benchmark('zebralogicbench', TaskConfig(model='mock', datasets=['zebralogicbench']))


def _sample(header):
    target = json.dumps({'header': header, 'rows': [['1', 'Alice', 'red']]})
    return Sample(input='q', target=target)


def test_house_header_builds_prompt_template():
    adapter = _adapter()
    prompt = adapter.format_prompt_template(_sample(['House', 'Name', 'Color']))
    assert prompt


@pytest.mark.parametrize('header', [['Name', 'House', 'Color'], ['house', 'Name', 'Color']])
def test_non_house_first_header_raises_value_error(header):
    adapter = _adapter()
    with pytest.raises(ValueError, match='first header column'):
        adapter.format_prompt_template(_sample(header))
