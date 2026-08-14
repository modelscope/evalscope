import pytest
from pydantic import ValidationError

from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricKind, MetricSemantics, ValueRange


def _accuracy_kwargs(**overrides) -> dict:
    kwargs = dict(
        semantic_id='quality.accuracy.ratio',
        metric_name='Accuracy',
        kind=MetricKind.QUALITY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=ValueRange(min=0.0, max=1.0),
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=1,
    )
    kwargs.update(overrides)
    return kwargs


def test_quality_and_diagnostic_semantics_are_valid() -> None:
    quality = MetricSemantics(**_accuracy_kwargs())
    diagnostic = MetricSemantics.diagnostic('Failed Requests')

    assert quality.raw_unit is None
    assert diagnostic.kind is MetricKind.DIAGNOSTIC
    assert diagnostic.direction is MetricDirection.NONE


@pytest.mark.parametrize(
    'overrides',
    [
        {'kind': MetricKind.QUALITY, 'direction': MetricDirection.NONE},
        {'kind': MetricKind.DIAGNOSTIC, 'direction': MetricDirection.HIGHER_IS_BETTER},
        {'value_range': None},
        {'display_multiplier': None},
        {'display_multiplier': 0.0},
        {'display_precision': -1},
    ],
)
def test_invalid_semantics_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        MetricSemantics(**_accuracy_kwargs(**overrides))


@pytest.mark.parametrize('bounds', [(1.0, 1.0), (2.0, 1.0), (0.0, float('inf')), (float('nan'), 1.0)])
def test_invalid_ranges_are_rejected(bounds: tuple) -> None:
    with pytest.raises(ValidationError):
        ValueRange(min=bounds[0], max=bounds[1])


def test_contract_is_frozen_and_forbids_extra_fields() -> None:
    semantics = MetricSemantics(**_accuracy_kwargs())
    with pytest.raises(ValidationError):
        semantics.display_precision = 3
    with pytest.raises(ValidationError):
        MetricSemantics(**_accuracy_kwargs(unknown_field='x'))
