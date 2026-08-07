"""Unit tests for ScreenSpot-Pro prediction parsing and scoring helpers.

The scoring path is a silent-failure risk: a mis-parsed coordinate still looks like a
plausible click point, so a regression would corrupt scores without raising anything.
"""
import re

from evalscope.benchmarks.screenspot_pro.screenspot_pro_adapter import PROMPT_TEMPLATE
from evalscope.benchmarks.screenspot_pro.utils import (
    _NUM_TOKEN,
    normalize_bbox,
    parse_point,
    point_in_bbox,
    to_normalized_point,
)

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


def test_point_in_bbox_is_strict() -> None:
    # powerpoint_windows_0: a 12x12 px icon on a 2880x1800 screen
    gt = normalize_bbox([2341, 1238, 2353, 1250], 2880, 1800)

    center = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
    assert point_in_bbox(center, gt)
    # Edges are inclusive
    assert point_in_bbox((gt[0], gt[1]), gt)
    # 1.4 px above the box top is a miss: the tolerance is the element itself
    assert not point_in_bbox((0.8140, 0.6870), gt)
