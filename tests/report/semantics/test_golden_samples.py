"""Golden-sample tests for the shared metric formatting contract.

Feature: metric-semantics-governance

``evalscope/metrics/semantics/golden_samples.json`` is read by this module and by the vitest
suite ``evalscope/web/src/domain/metric/goldenSamples.test.ts``, so both implementations of the
formatting rules are pinned to the same expected strings (requirements 13.9, 20.2).
"""

import json
import pytest
from typing import Any, Dict, List

from evalscope.api.metric.semantics import MetricDisplayKind, MetricSemantics
from evalscope.metrics.semantics.formatting import (
    GOLDEN_SAMPLES_PATH,
    MISSING_PLACEHOLDER,
    GoldenSample,
    format_metric_value,
    format_raw_metric_value,
    load_golden_samples,
)

#: Keys that form the assertion contract shared with the frontend.
CONTRACT_KEYS = frozenset({'semantics', 'value', 'expected_primary', 'expected_raw'})

#: Metadata keys consumers may ignore.
METADATA_KEYS = frozenset({'id', 'description'})

SAMPLES: List[GoldenSample] = load_golden_samples()


def load_raw_samples() -> List[Dict[str, Any]]:
    """Read the golden samples file without model validation."""
    with open(GOLDEN_SAMPLES_PATH, 'r', encoding='utf-8') as stream:
        return json.load(stream)


def sample_ids() -> List[str]:
    """Identifiers used as pytest parameter ids."""
    return [sample.id for sample in SAMPLES]


class TestGoldenSampleFile:
    """The file itself must stay a consumable, self-describing contract."""

    def test_file_is_packaged_next_to_the_formatter(self) -> None:
        assert GOLDEN_SAMPLES_PATH.name == 'golden_samples.json'
        assert GOLDEN_SAMPLES_PATH.is_file()

    def test_top_level_is_a_non_empty_array(self) -> None:
        raw = load_raw_samples()
        assert isinstance(raw, list)
        assert len(raw) >= 7

    def test_every_sample_declares_the_contract_keys(self) -> None:
        for raw in load_raw_samples():
            assert CONTRACT_KEYS.issubset(raw), f"sample {raw.get('id')} misses contract keys"
            assert set(raw) <= CONTRACT_KEYS | METADATA_KEYS, f"sample {raw.get('id')} has unknown keys"

    def test_sample_ids_are_unique(self) -> None:
        ids = sample_ids()
        assert len(set(ids)) == len(ids)

    def test_semantics_payloads_are_json_dumps_of_the_contract(self) -> None:
        """The payload must equal ``MetricSemantics.model_dump(mode='json')`` so TS can consume it."""
        for raw in load_raw_samples():
            payload = raw['semantics']
            if payload is None:
                continue
            assert payload == MetricSemantics(**payload).model_dump(mode='json'), f"sample {raw['id']}"

    def test_expected_texts_are_non_empty_strings(self) -> None:
        for sample in SAMPLES:
            assert isinstance(sample.expected_primary, str) and sample.expected_primary
            assert isinstance(sample.expected_raw, str) and sample.expected_raw


class TestGoldenSampleCoverage:
    """The samples must exercise every branch the two implementations share."""

    def test_covers_percent_ratio_and_official_scale(self) -> None:
        percent = [s for s in SAMPLES if s.semantics and s.semantics.display_kind == MetricDisplayKind.PERCENT]
        multipliers = {s.semantics.display_multiplier for s in percent}
        assert 100.0 in multipliers, 'no [0,1] ratio rendered as percent'
        assert 1.0 in multipliers, 'no official 0-100 scale sample'

    @pytest.mark.parametrize('unit', ['s', 'ms'])
    def test_covers_time_units(self, unit: str) -> None:
        units = {s.semantics.display_unit for s in SAMPLES if s.semantics}
        assert unit in units

    def test_covers_unitless_number(self) -> None:
        assert any(
            s.semantics is not None and s.semantics.display_kind == MetricDisplayKind.NUMBER
            and s.semantics.display_unit is None for s in SAMPLES
        )

    def test_covers_missing_value(self) -> None:
        missing = [s for s in SAMPLES if s.value is None and s.semantics is not None]
        assert missing, 'no missing value sample with semantics'
        assert all(s.expected_primary == MISSING_PLACEHOLDER for s in missing)

    def test_covers_diagnostic_fallback(self) -> None:
        fallback = [s for s in SAMPLES if s.semantics is None and s.value is not None]
        assert fallback, 'no diagnostic fallback sample'


@pytest.mark.parametrize('sample', SAMPLES, ids=sample_ids())
def test_backend_matches_golden_primary_text(sample: GoldenSample) -> None:
    """Requirements 13.9, 20.2: the backend primary text equals the shared expectation."""
    assert format_metric_value(sample.value, sample.semantics) == sample.expected_primary


@pytest.mark.parametrize('sample', SAMPLES, ids=sample_ids())
def test_backend_matches_golden_raw_text(sample: GoldenSample) -> None:
    """Requirements 13.9, 20.2: the backend raw text equals the shared expectation."""
    assert format_raw_metric_value(sample.value, sample.semantics) == sample.expected_raw
