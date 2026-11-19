"""Unit tests for ScreenSpot-Pro prediction parsing and scoring helpers.

The scoring path is a silent-failure risk: a mis-parsed coordinate still looks like a
plausible click point, so a regression would corrupt scores without raising anything.
"""
import json
import re

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.screenspot_pro.screenspot_pro_adapter import PROMPT_TEMPLATE
from evalscope.benchmarks.screenspot_pro.utils import (
    _NUM_TOKEN,
    normalize_bbox,
    parse_point,
    point_in_bbox,
    to_normalized_point,
)
from evalscope.config import TaskConfig

_POINT_RE = rf'[\[\(]\s*({_NUM_TOKEN})\s*(?:,|\s)\s*({_NUM_TOKEN})\s*[\]\)]'


def test_prompt_contains_no_parseable_coordinate_pair() -> None:
    """The prompt must not contain a number pair that the parser could mistake for an answer.

    An earlier prompt said 'within [0, 1]'; models restating it had '[0, 1]' scored as
    their prediction.
    """
    prompt = PROMPT_TEMPLATE.format(instruction='Mark dimensions')
    assert re.findall(_POINT_RE, prompt) == []
    assert 'Answer:' in prompt


def test_parse_point_prefers_answer_marker() -> None:
    assert parse_point('Answer: [0.31, 0.42]') == (0.31, 0.42)
    assert parse_point('**Final Answer:** [0.12, 0.34]') == (0.12, 0.34)
    assert parse_point('answer: (0.11, 0.22)') == (0.11, 0.22)
    # Marker and answer separated by a newline
    assert parse_point('Reasoning ...\nAnswer:\n[0.25, 0.15]') == (0.25, 0.15)


def test_parse_point_ignores_reasoning_around_the_answer() -> None:
    """Neither preceding reasoning nor trailing prose may shadow the answer line."""
    leading = 'The grid spans (0, 0) to (1, 1) and the toolbar is 200px wide.\nAnswer: [0.31, 0.42]'
    assert parse_point(leading) == (0.31, 0.42)

    trailing = 'Answer: [0.3, 0.07]\n(note: coordinates run from (0, 0) to (1, 1))'
    assert parse_point(trailing) == (0.3, 0.07)


def test_parse_point_falls_back_when_format_ignored() -> None:
    # No marker at all: structured patterns are still trusted, taking the last match
    assert parse_point('I reason ... so the click point is [0.6, 0.2]') == (0.6, 0.2)
    # Marker present but carries no coordinate
    assert parse_point('The point is [0.4, 0.5]. Answer: I cannot tell') == (0.4, 0.5)


def test_loose_formats_only_trusted_on_the_answer_line() -> None:
    """Loose notation must not be harvested from free prose.

    A truncated reasoning trace describes layout ('x=175 to x=935, y=85') and uses
    ordinals ('the 5th or 6th icon'); reading those as the answer fabricates a
    confident-looking click point for a model that never answered.
    """
    # On an answer line these formats are legitimate
    assert parse_point('Answer: 0.3 0.07') == (0.3, 0.07)
    assert parse_point('Answer: x=0.2, y=0.9') == (0.2, 0.9)

    # In free prose without a marker they must be rejected rather than invented
    bounds = 'The Stata window is positioned roughly from x=175 to x=935, y=85 to'
    assert parse_point(bounds) is None

    ordinals = 'the Output Port tool is the 5th or 6th icon from the left in that toolbar segment'
    assert parse_point(ordinals) is None

    # An explicit point pair in prose is still unambiguous enough to accept
    assert parse_point('the zoom icon sits at (400, 135) in the toolbar') == (400.0, 135.0)


def test_parse_point_accepted_formats() -> None:
    assert parse_point('Answer: [814, 687]') == (814.0, 687.0)
    assert parse_point('Answer: [50%, 25%]') == (0.5, 0.25)
    assert parse_point('Answer: x=0.2, y=0.9') == (0.2, 0.9)
    # A bounding box is reduced to its center
    assert parse_point('Answer: <bbox>10, 20, 30, 40</bbox>') == (20.0, 30.0)
    assert parse_point('I am unable to locate this element.') is None


def test_to_normalized_point_infers_convention_by_magnitude() -> None:
    size = (2880, 1800)
    # values in [0, 1] are already normalized
    assert to_normalized_point((0.8, 0.4), size) == (0.8, 0.4)
    # values up to 1000 are the thousandths grid
    assert to_normalized_point((805, 391), size) == (0.805, 0.391)
    # larger values are pixels of the image sent
    assert to_normalized_point((2320, 704), size) == (2320 / 2880, 704 / 1800)


def test_to_normalized_point_explicit_format_skips_inference() -> None:
    """Magnitude cannot tell pixels from thousandths below 1000, so a fixed convention must win.

    A pixel answer in the top-left 1000x1000 px (menus, toolbars) is read as thousandths by
    ``auto``; so is every pixel answer on an image downscaled to 1000 px or less.
    """
    size = (2560, 1440)
    assert to_normalized_point((500, 300), size) == (0.5, 0.3)
    assert to_normalized_point((500, 300), size, 'pixel') == (500 / 2560, 300 / 1440)
    assert to_normalized_point((480, 270), (960, 540), 'pixel') == (0.5, 0.5)

    assert to_normalized_point((805, 391), size, 'thousandths') == (0.805, 0.391)
    assert to_normalized_point((1.0, 0.5), size, 'thousandths') == (0.001, 0.0005)
    assert to_normalized_point((0.8, 0.4), size, 'normalized') == (0.8, 0.4)


def test_extract_answer_reads_coordinate_format_from_extra_params() -> None:
    def extract(dataset_args: dict) -> str:
        config = TaskConfig(model='mock', datasets=['screenspot_pro'], dataset_args={'screenspot_pro': dataset_args})
        adapter = get_benchmark('screenspot_pro', config)
        state = TaskState(
            model='mock',
            sample=Sample(input='q', target='[0, 0, 1, 1]', metadata={'sent_size': [2560, 1440]}),
            output=ModelOutput.from_content('mock', 'Answer: [500, 300]'),
        )
        return adapter.extract_answer('Answer: [500, 300]', state)

    assert json.loads(extract({})) == [0.5, 0.3]
    assert json.loads(extract({'extra_params': {'coordinate_format': 'pixel'}})) == [500 / 2560, 300 / 1440]
    with pytest.raises(ValueError):
        extract({'extra_params': {'coordinate_format': 'pixels'}})


def test_point_in_bbox_is_strict() -> None:
    # powerpoint_windows_0: a 12x12 px icon on a 2880x1800 screen
    gt = normalize_bbox([2341, 1238, 2353, 1250], 2880, 1800)

    center = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
    assert point_in_bbox(center, gt)
    # Edges are inclusive
    assert point_in_bbox((gt[0], gt[1]), gt)
    # 1.4 px above the box top is a miss: the tolerance is the element itself
    assert not point_in_bbox((0.8140, 0.6870), gt)


@pytest.mark.parametrize(
    ('coordinate_format', 'answer', 'bbox', 'expected'),
    [
        ('pixel', 'Answer: [2341, 1238]', normalize_bbox([2341, 1238, 2353, 1250], 2880, 1800), 1.0),
        ('normalized', 'Answer: [0.4999999, 0.6]', [0.5, 0.5, 0.7, 0.7], 0.0),
        ('normalized', 'Answer: [0.0000001, 0.6]', [0.0000001, 0.5, 0.7, 0.7], 1.0),
        ('normalized', 'Cannot locate the element.', [0, 0, 1, 1], 0.0),
    ],
)
def test_scores_clicks_at_full_precision(
    coordinate_format: str, answer: str, bbox: list[float], expected: float
) -> None:
    config = TaskConfig(
        datasets=['screenspot_pro'],
        dataset_args={'screenspot_pro': {'extra_params': {'coordinate_format': coordinate_format}}},
    )
    adapter = get_benchmark('screenspot_pro', config)
    sample = Sample(
        input='q', target=str(bbox), metadata={'sent_size': [2880, 1800], 'bbox_norm': bbox, 'ui_type': 'icon'}
    )
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content('mock', answer), completed=True)
    assert adapter.calculate_metrics(state).score.value == {'acc': expected, 'icon_acc': expected}
