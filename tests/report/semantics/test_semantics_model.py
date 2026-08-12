"""Unit and property tests for the MetricSemantics contract model."""
import pytest
import strategies
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from typing import Any, Dict, List, Tuple

from evalscope.api.metric.semantics import (
    METRIC_CONTRACT_VERSION,
    MetricDirection,
    MetricDisplayKind,
    MetricRole,
    MetricSemantics,
    ValueRange,
)


def _accuracy_kwargs(**overrides) -> dict:
    kwargs = dict(
        semantic_id='quality.accuracy.ratio',
        metric_name='Accuracy',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=ValueRange(min=0.0, max=1.0),
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=1,
    )
    kwargs.update(overrides)
    return kwargs


class TestMetricSemanticsValid:

    def test_percent_primary_metric_is_accepted(self) -> None:
        semantics = MetricSemantics(**_accuracy_kwargs())

        assert semantics.contract_version == METRIC_CONTRACT_VERSION == 1
        assert semantics.raw_unit is None

    def test_diagnostic_metric_without_direction_is_accepted(self) -> None:
        semantics = MetricSemantics(
            semantic_id='diagnostic.count.items',
            metric_name='Failed Requests',
            role=MetricRole.DIAGNOSTIC,
            direction=MetricDirection.NONE,
        )

        assert semantics.display_kind == MetricDisplayKind.NUMBER
        assert semantics.display_precision == 4

    def test_model_is_frozen_and_forbids_extra_fields(self) -> None:
        semantics = MetricSemantics(**_accuracy_kwargs())

        with pytest.raises(ValidationError):
            semantics.display_precision = 3

        with pytest.raises(ValidationError):
            MetricSemantics(**_accuracy_kwargs(unknown_field='x'))


class TestMetricSemanticsValidation:
    """Rules with no property counterpart.

    The role / direction pair, the percent display bundle and the closed enum domains are covered
    exhaustively by ``TestMetricSemanticsProperties`` below, which asserts the same rejections plus
    the error message contents over generated inputs. Only the rules those properties do not
    generate are pinned here.
    """

    @pytest.mark.parametrize('bounds', [(1.0, 1.0), (2.0, 1.0), (0.0, float('inf')), (float('nan'), 1.0)])
    def test_rejects_invalid_value_range(self, bounds: tuple) -> None:
        minimum, maximum = bounds
        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**_accuracy_kwargs(value_range={'min': minimum, 'max': maximum}))

        assert 'value_range' in str(excinfo.value)

    @pytest.mark.parametrize('multiplier', [0.0, -1.0, float('inf'), float('nan')])
    def test_rejects_invalid_multiplier(self, multiplier: float) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**_accuracy_kwargs(display_multiplier=multiplier))

        message = str(excinfo.value)
        assert 'quality.accuracy.ratio' in message
        assert 'display_multiplier' in message

    def test_rejects_negative_precision(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**_accuracy_kwargs(display_precision=-1))

        message = str(excinfo.value)
        assert 'quality.accuracy.ratio' in message
        assert 'display_precision' in message


class TestMetricSemanticsProperties:
    """Property based tests of the contract level rules."""

    @given(kwargs=strategies.role_direction_kwargs())
    def test_role_and_direction_are_consistent(self, kwargs: Dict[str, Any]) -> None:
        """Verify role and direction consistency.

        For any MetricSemantics field combination, construction succeeds if and only if
        (role in {primary, auxiliary} and direction != none) or
        (role == diagnostic and direction == none); on failure the error message contains
        both semantic_id and metric_name.
        """
        role: MetricRole = kwargs['role']
        direction: MetricDirection = kwargs['direction']

        if strategies.is_role_direction_consistent(role, direction):
            semantics = MetricSemantics(**kwargs)

            assert semantics.role == role
            assert semantics.direction == direction
            return

        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**kwargs)

        message = str(excinfo.value)
        assert kwargs['semantic_id'] in message
        assert kwargs['metric_name'] in message

    @given(kwargs=strategies.valid_semantics_kwargs())
    def test_enum_domain_values_are_accepted(self, kwargs: Dict[str, Any]) -> None:
        """Verify that values inside the closed enum domains are accepted.

        Accepting half of the property: any value inside the three closed enum domains
        (role, direction, display_kind) builds a MetricSemantics, whether it is passed as
        the enum member or as its raw string value.
        """
        from_members = MetricSemantics(**kwargs)

        assert from_members.role in set(MetricRole)
        assert from_members.direction in set(MetricDirection)
        assert from_members.display_kind in set(MetricDisplayKind)

        raw_kwargs = dict(kwargs)
        for field_name in ('role', 'direction', 'display_kind'):
            raw_kwargs[field_name] = kwargs[field_name].value
        from_raw_values = MetricSemantics(**raw_kwargs)

        assert from_raw_values == from_members

    @given(case=strategies.invalid_enum_kwargs())
    def test_out_of_domain_enum_values_are_rejected(self, case: Tuple[Dict[str, Any], str]) -> None:
        """Verify that values outside the closed enum domains are rejected.

        Rejecting half of the property: for any role, direction or display_kind value outside
        its legal enum set, constructing MetricSemantics fails and the reported error points at
        that field.
        """
        kwargs, field_name = case

        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**kwargs)

        assert any(error['loc'] == (field_name, ) for error in excinfo.value.errors())

    @given(kwargs=strategies.valid_semantics_kwargs(display_kinds=st.just(MetricDisplayKind.PERCENT)))
    def test_percent_with_range_and_multiplier_is_accepted(self, kwargs: Dict[str, Any]) -> None:
        """Verify that percent declarations with the required display fields are accepted.

        Accepting half of the property: for any percent declaration carrying both value_range
        and display_multiplier, construction succeeds and both fields survive unchanged.
        """
        semantics = MetricSemantics(**kwargs)

        assert semantics.display_kind == MetricDisplayKind.PERCENT
        assert semantics.value_range == kwargs['value_range']
        assert semantics.display_multiplier == kwargs['display_multiplier']

    @given(case=strategies.percent_missing_display_field_kwargs())
    def test_percent_missing_display_fields_is_rejected(self, case: Tuple[Dict[str, Any], List[str]]) -> None:
        """Verify that percent declarations missing display fields are rejected.

        Rejecting half of the property: for any percent declaration missing value_range,
        display_multiplier or both, construction fails and the error message names the
        semantic_id together with every missing field.
        """
        kwargs, dropped = case
        assert kwargs['display_kind'] == MetricDisplayKind.PERCENT

        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**kwargs)

        message = str(excinfo.value)
        assert kwargs['semantic_id'] in message
        for field_name in dropped:
            assert field_name in message

    @given(case=strategies.percent_missing_display_field_kwargs())
    def test_number_display_kind_is_not_subject_to_the_percent_rule(
        self, case: Tuple[Dict[str, Any], List[str]]
    ) -> None:
        """Verify that number declarations are not subject to the percent display rule.

        Scope of the property: the very same declarations become valid once display_kind is
        'number', so the rule applies to percent display only.
        """
        kwargs, _ = case
        kwargs['display_kind'] = MetricDisplayKind.NUMBER

        semantics = MetricSemantics(**kwargs)

        assert semantics.display_kind == MetricDisplayKind.NUMBER
        assert semantics.value_range == kwargs['value_range']
        assert semantics.display_multiplier == kwargs['display_multiplier']
