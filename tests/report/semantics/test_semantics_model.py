"""Unit and property tests for the MetricSemantics contract model.

Feature: metric-semantics-governance
"""
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
        comparison_group='quality.accuracy',
    )
    kwargs.update(overrides)
    return kwargs


class TestMetricSemanticsValid:

    def test_percent_primary_metric_is_accepted(self) -> None:
        semantics = MetricSemantics(**_accuracy_kwargs())

        assert semantics.contract_version == METRIC_CONTRACT_VERSION == 1
        assert semantics.aggregation_group is None
        assert semantics.raw_unit is None

    def test_diagnostic_metric_without_direction_is_accepted(self) -> None:
        semantics = MetricSemantics(
            semantic_id='diagnostic.count.items',
            metric_name='Failed Requests',
            role=MetricRole.DIAGNOSTIC,
            direction=MetricDirection.NONE,
        )

        assert semantics.comparison_group is None
        assert semantics.display_kind == MetricDisplayKind.NUMBER
        assert semantics.display_precision == 4

    def test_model_is_frozen_and_forbids_extra_fields(self) -> None:
        semantics = MetricSemantics(**_accuracy_kwargs())

        with pytest.raises(ValidationError):
            semantics.display_precision = 3

        with pytest.raises(ValidationError):
            MetricSemantics(**_accuracy_kwargs(unknown_field='x'))


class TestMetricSemanticsValidation:

    @pytest.mark.parametrize('role', [MetricRole.PRIMARY, MetricRole.AUXILIARY])
    def test_r1_scored_role_requires_direction(self, role: MetricRole) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**_accuracy_kwargs(role=role, direction=MetricDirection.NONE))

        message = str(excinfo.value)
        assert 'quality.accuracy.ratio' in message
        assert 'Accuracy' in message

    def test_r2_diagnostic_rejects_direction(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(
                semantic_id='diagnostic.count.items',
                metric_name='Steps',
                role=MetricRole.DIAGNOSTIC,
                direction=MetricDirection.LOWER_IS_BETTER,
            )

        message = str(excinfo.value)
        assert 'diagnostic.count.items' in message
        assert 'Steps' in message

    def test_r3_diagnostic_rejects_comparison_group(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(
                semantic_id='diagnostic.count.items',
                metric_name='Steps',
                role=MetricRole.DIAGNOSTIC,
                direction=MetricDirection.NONE,
                comparison_group='diagnostic.count',
            )

        assert 'diagnostic.count.items' in str(excinfo.value)

    @pytest.mark.parametrize('missing', ['value_range', 'display_multiplier'])
    def test_r4_percent_requires_range_and_multiplier(self, missing: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**_accuracy_kwargs(**{missing: None}))

        message = str(excinfo.value)
        assert 'quality.accuracy.ratio' in message
        assert missing in message

    @pytest.mark.parametrize('bounds', [(1.0, 1.0), (2.0, 1.0), (0.0, float('inf')), (float('nan'), 1.0)])
    def test_r5_rejects_invalid_value_range(self, bounds: tuple) -> None:
        minimum, maximum = bounds
        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**_accuracy_kwargs(value_range={'min': minimum, 'max': maximum}))

        assert 'value_range' in str(excinfo.value)

    @pytest.mark.parametrize('multiplier', [0.0, -1.0, float('inf'), float('nan')])
    def test_r5_rejects_invalid_multiplier(self, multiplier: float) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**_accuracy_kwargs(display_multiplier=multiplier))

        message = str(excinfo.value)
        assert 'quality.accuracy.ratio' in message
        assert 'display_multiplier' in message

    def test_r5_rejects_negative_precision(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**_accuracy_kwargs(display_precision=-1))

        message = str(excinfo.value)
        assert 'quality.accuracy.ratio' in message
        assert 'display_precision' in message

    @pytest.mark.parametrize('field, value', [('role', 'unknown'), ('direction', 'up'), ('display_kind', 'ratio')])
    def test_enum_domains_are_closed(self, field: str, value: str) -> None:
        with pytest.raises(ValidationError):
            MetricSemantics(**_accuracy_kwargs(**{field: value}))


class TestMetricSemanticsProperties:
    """Property based tests of the contract level rules."""

    @given(kwargs=strategies.role_direction_kwargs())
    def test_role_and_direction_are_consistent(self, kwargs: Dict[str, Any]) -> None:
        """Feature: metric-semantics-governance, Property 1: 角色与方向一致性.

        For any MetricSemantics field combination, construction succeeds if and only if
        (role in {primary, auxiliary} and direction != none) or
        (role == diagnostic and direction == none); on failure the error message contains
        both semantic_id and metric_name.

        **Validates: Requirements 1.3, 1.4**
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
        """Feature: metric-semantics-governance, Property 2: 枚举取值域封闭.

        Accepting half of the property: any value inside the three closed enum domains
        (role, direction, display_kind) builds a MetricSemantics, whether it is passed as
        the enum member or as its raw string value.

        **Validates: Requirements 1.2**
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
        """Feature: metric-semantics-governance, Property 2: 枚举取值域封闭.

        Rejecting half of the property: for any role, direction or display_kind value outside
        its legal enum set, constructing MetricSemantics fails and the reported error points at
        that field.

        **Validates: Requirements 1.2**
        """
        kwargs, field_name = case

        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**kwargs)

        assert any(error['loc'] == (field_name, ) for error in excinfo.value.errors())

    @given(kwargs=strategies.diagnostic_with_comparison_group_kwargs())
    def test_diagnostic_metric_rejects_any_comparison_group(self, kwargs: Dict[str, Any]) -> None:
        """Feature: metric-semantics-governance, Property 3: 诊断指标不参与比较分组.

        Rejecting half of the property: for any semantics declaration with role == diagnostic and
        any non-empty comparison_group string, construction fails and the error message contains
        the semantic_id.

        **Validates: Requirements 1.5**
        """
        assert kwargs['role'] == MetricRole.DIAGNOSTIC
        assert kwargs['comparison_group']

        with pytest.raises(ValidationError) as excinfo:
            MetricSemantics(**kwargs)

        message = str(excinfo.value)
        assert kwargs['semantic_id'] in message
        assert 'comparison_group' in message

    @given(kwargs=strategies.valid_semantics_kwargs(roles=st.just(MetricRole.DIAGNOSTIC)))
    def test_diagnostic_metric_without_comparison_group_is_accepted(self, kwargs: Dict[str, Any]) -> None:
        """Feature: metric-semantics-governance, Property 3: 诊断指标不参与比较分组.

        Accepting half of the property: the same diagnostic declaration with comparison_group
        left empty always builds, so only the comparison group triggers the rejection.

        **Validates: Requirements 1.5**
        """
        kwargs['comparison_group'] = None

        semantics = MetricSemantics(**kwargs)

        assert semantics.role == MetricRole.DIAGNOSTIC
        assert semantics.comparison_group is None

    @given(kwargs=strategies.valid_semantics_kwargs(display_kinds=st.just(MetricDisplayKind.PERCENT)))
    def test_percent_with_range_and_multiplier_is_accepted(self, kwargs: Dict[str, Any]) -> None:
        """Feature: metric-semantics-governance, Property 4: 百分比展示必填字段.

        Accepting half of the property: for any percent declaration carrying both value_range
        and display_multiplier, construction succeeds and both fields survive unchanged.

        **Validates: Requirements 1.6**
        """
        semantics = MetricSemantics(**kwargs)

        assert semantics.display_kind == MetricDisplayKind.PERCENT
        assert semantics.value_range == kwargs['value_range']
        assert semantics.display_multiplier == kwargs['display_multiplier']

    @given(case=strategies.percent_missing_display_field_kwargs())
    def test_percent_missing_display_fields_is_rejected(self, case: Tuple[Dict[str, Any], List[str]]) -> None:
        """Feature: metric-semantics-governance, Property 4: 百分比展示必填字段.

        Rejecting half of the property: for any percent declaration missing value_range,
        display_multiplier or both, construction fails and the error message names the
        semantic_id together with every missing field.

        **Validates: Requirements 1.6**
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
        """Feature: metric-semantics-governance, Property 4: 百分比展示必填字段.

        Scope of the property: the very same declarations become valid once display_kind is
        'number', so the rule applies to percent display only.

        **Validates: Requirements 1.6**
        """
        kwargs, _ = case
        kwargs['display_kind'] = MetricDisplayKind.NUMBER

        semantics = MetricSemantics(**kwargs)

        assert semantics.display_kind == MetricDisplayKind.NUMBER
        assert semantics.value_range == kwargs['value_range']
        assert semantics.display_multiplier == kwargs['display_multiplier']
