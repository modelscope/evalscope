"""Golden-sample tests for the shared metric formatting contract.

``tests/report/semantics/golden_samples.json`` is read by this module and by the vitest suite
``evalscope/web/src/domain/metric/metricFormat.test.ts``, so both implementations of the formatting
rules are pinned to the same expected strings.

The sample model and its loader live here rather than in the shipped package: they are test
scaffolding, and the only production consumer of the contract is ``format_metric_value``.
``expected_raw`` pins the frontend's ``FormattedMetric.raw`` (it backs the value tooltips); no
backend function produces the unscaled text, so this module does not assert it. ``expected_label``
is asserted on both sides: ``format_metric_label`` and the frontend ``formatMetricIdentityLabel``
are a second pair of parallel implementations, and nothing else pins them to each other.
"""

import json
import pytest
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional

from evalscope.api.metric.semantics import MetricDisplayKind, MetricIdentity, MetricSemantics
from evalscope.metrics.semantics.formatting import MISSING_PLACEHOLDER, format_metric_label, format_metric_value

#: Location of the golden samples shared with the frontend formatting tests.
GOLDEN_SAMPLES_PATH = Path(__file__).with_name('golden_samples.json')


class GoldenSample(BaseModel):
    """One entry of ``golden_samples.json``, the backend/frontend formatting contract."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    id: str
    """Stable identifier, unique inside the file."""

    description: str = Field(default='')
    """Human readable note. Not part of the assertion contract."""

    semantics: Optional[MetricSemantics] = Field(default=None)
    """Semantics payload, or ``None`` to exercise the diagnostic fallback."""

    identity: Optional[MetricIdentity] = Field(default=None)
    """Identity the label is rendered from. ``None`` means the sample only pins value formatting."""

    legacy_name: Optional[str] = Field(default=None)
    """Original v1 spelling a diagnostic label falls back to."""

    value: Optional[float] = Field(default=None)
    """Stored metric value, or ``None`` to exercise the missing value path."""

    expected_primary: str
    """Expected ``format_metric_value`` output."""

    expected_raw: str
    """Expected ``FormattedMetric.raw`` of the frontend primitive."""

    expected_label: Optional[str] = Field(default=None)
    """Expected label of ``identity``, shared with the frontend ``formatMetricIdentityLabel``."""


def load_golden_samples() -> List[GoldenSample]:
    """Load and validate the shared formatting golden samples, in file order."""
    with open(GOLDEN_SAMPLES_PATH, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    return [GoldenSample(**sample) for sample in payload]


SAMPLES: List[GoldenSample] = load_golden_samples()


def load_raw_samples() -> List[Dict[str, Any]]:
    """Read the golden samples file without model validation."""
    with open(GOLDEN_SAMPLES_PATH, 'r', encoding='utf-8') as stream:
        return json.load(stream)


def sample_ids() -> List[str]:
    """Identifiers used as pytest parameter ids."""
    return [sample.id for sample in SAMPLES]


class TestGoldenSampleFile:
    """The file itself must stay a consumable, self-describing contract.

    ``SAMPLES`` is loaded through ``GoldenSample`` (``extra='forbid'``, required fields) at module
    import, so a missing file, a malformed entry or an unknown key already errors at collection.
    Only the invariants pydantic does *not* cover are asserted here.
    """

    def test_top_level_is_a_non_empty_array(self) -> None:
        assert len(load_raw_samples()) >= 7

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

    def test_label_samples_declare_an_identity(self) -> None:
        """``expected_label`` is meaningless without the identity it is rendered from."""
        for sample in SAMPLES:
            assert (sample.expected_label is None) == (sample.identity is None), f'sample {sample.id}'

    def test_identity_payloads_are_json_dumps_of_the_contract(self) -> None:
        """The payload must equal ``MetricIdentity.model_dump(mode='json')`` so TS can consume it."""
        for raw in load_raw_samples():
            payload = raw.get('identity')
            if payload is None:
                continue
            assert payload == MetricIdentity(**payload).model_dump(mode='json'), f"sample {raw['id']}"


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

    def test_covers_label_shapes(self) -> None:
        """The label path needs its own coverage: the value samples do not exercise it."""
        labelled = [s for s in SAMPLES if s.identity is not None and s.expected_label is not None]
        assert labelled, 'no label sample'
        assert any(not s.identity.dimensions for s in labelled), 'no dimensionless label'
        assert any(len(s.identity.dimensions) >= 2 for s in labelled), 'no multi-dimension label'
        assert any('↓' in s.expected_label for s in labelled), 'no lower_is_better label'
        assert any(
            isinstance(value, bool) for s in labelled for value in s.identity.dimensions.values()
        ), 'no boolean dimension label'
        assert any(s.legacy_name for s in labelled), 'no diagnostic label falling back to a legacy name'
        assert any(
            isinstance(value, str) and value != value.lower()
            for s in labelled
            for value in s.identity.dimensions.values()
        ), 'no dimension value with upper-case letters, so acronym casing is unpinned'


@pytest.mark.parametrize(
    'sample',
    [s for s in SAMPLES if s.expected_label is not None],
    ids=[s.id for s in SAMPLES if s.expected_label is not None],
)
def test_backend_matches_golden_label(sample: GoldenSample) -> None:
    """The backend label equals the shared expectation, so it cannot drift from the frontend."""
    assert format_metric_label(sample.identity, sample.semantics, sample.legacy_name) == sample.expected_label


@pytest.mark.parametrize('sample', SAMPLES, ids=sample_ids())
def test_backend_matches_golden_primary_text(sample: GoldenSample) -> None:
    """The backend display text equals the shared expectation."""
    assert format_metric_value(sample.value, sample.semantics) == sample.expected_primary
