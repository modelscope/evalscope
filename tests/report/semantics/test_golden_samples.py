"""Golden-sample tests for the shared metric formatting contract.

``tests/report/semantics/golden_samples.json`` is read by this module and by the vitest suite
``evalscope/web/src/domain/metric/metricFormat.test.ts``. Semantics payloads are declared once in
its registry and samples refer to them by key, so both implementations stay pinned to the same
wire contracts and expected strings without repeating each contract in every case.

The sample model and its loader live here rather than in the shipped package: they are test
scaffolding, and the only production consumer of the contract is ``format_metric_value``.
``expected_raw`` pins the frontend's ``FormattedMetric.raw`` (it backs the value tooltips); no
backend function produces the unscaled text, so this module does not assert it. ``expected_label``
is asserted on both sides: ``format_metric_label`` and the frontend ``formatMetricIdentityLabel``
are a second pair of parallel implementations, and nothing else pins them to each other.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

from evalscope.api.metric.semantics import MetricDisplayKind, MetricIdentity, MetricSemantics
from evalscope.metrics.semantics.formatting import MISSING_PLACEHOLDER, format_metric_label, format_metric_value

#: Location of the golden samples shared with the frontend formatting tests.
GOLDEN_SAMPLES_PATH = Path(__file__).with_name('golden_samples.json')


class GoldenSampleSpec(BaseModel):
    """One compact sample entry before its semantics reference is resolved."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    id: str
    semantics_ref: Optional[str] = Field(default=None)

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


class GoldenFixture(BaseModel):
    """Shared semantics registry and compact formatting cases."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    semantics: Dict[str, MetricSemantics]
    samples: List[GoldenSampleSpec]


class GoldenSample(BaseModel):
    """Materialized sample consumed by the backend assertions."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    id: str
    semantics: Optional[MetricSemantics] = None
    identity: Optional[MetricIdentity] = None
    legacy_name: Optional[str] = None
    value: Optional[float] = None
    expected_primary: str
    expected_raw: str
    expected_label: Optional[str] = None


def load_golden_samples() -> List[GoldenSample]:
    """Load and validate the shared formatting golden samples, in file order."""
    with open(GOLDEN_SAMPLES_PATH, 'r', encoding='utf-8') as stream:
        fixture = GoldenFixture(**json.load(stream))
    return [
        GoldenSample(
            **sample.model_dump(exclude={'semantics_ref'}),
            semantics=fixture.semantics[sample.semantics_ref] if sample.semantics_ref else None,
        ) for sample in fixture.samples
    ]


SAMPLES: List[GoldenSample] = load_golden_samples()


def load_raw_fixture() -> Dict[str, Any]:
    """Read the golden fixture without model validation."""
    with open(GOLDEN_SAMPLES_PATH, 'r', encoding='utf-8') as stream:
        return json.load(stream)


def sample_ids() -> List[str]:
    """Identifiers used as pytest parameter ids."""
    return [sample.id for sample in SAMPLES]


class TestGoldenSampleFile:
    """The file itself must stay a consumable, self-describing contract.

    ``SAMPLES`` is loaded through ``GoldenSample`` (``extra='forbid'``, required fields) at module
    import, so a missing file, a malformed entry, an unknown key or a wrongly typed field already
    errors at collection. Only the invariants pydantic does *not* cover are asserted here.
    """

    def test_sample_ids_are_unique(self) -> None:
        ids = sample_ids()
        assert len(set(ids)) == len(ids)

    def test_semantics_payloads_are_json_dumps_of_the_contract(self) -> None:
        """The payload must equal ``MetricSemantics.model_dump(mode='json')`` so TS can consume it."""
        for key, payload in load_raw_fixture()['semantics'].items():
            assert payload == MetricSemantics(**payload).model_dump(mode='json'), f'semantics {key}'

    def test_label_samples_declare_an_identity(self) -> None:
        """``expected_label`` is meaningless without the identity it is rendered from."""
        for sample in SAMPLES:
            assert (sample.expected_label is None) == (sample.identity is None), f'sample {sample.id}'

    def test_identity_payloads_are_json_dumps_of_the_contract(self) -> None:
        """The payload must equal ``MetricIdentity.model_dump(mode='json')`` so TS can consume it."""
        for raw in load_raw_fixture()['samples']:
            payload = raw.get('identity')
            if payload is None:
                continue
            assert payload == MetricIdentity(**payload).model_dump(mode='json'), f"sample {raw['id']}"


class TestGoldenSampleCoverage:
    """The samples must exercise every branch the two implementations share.

    One assertion per branch of ``format_metric_value`` / ``format_metric_label``: without these the
    file could shrink to a single percent sample and both suites would still pass while silently
    covering nothing. Shapes pydantic already enforces are not re-checked.
    """

    def test_covers_every_value_rendering_branch(self) -> None:
        with_semantics = [s for s in SAMPLES if s.semantics]
        multipliers = {
            s.semantics.display_multiplier
            for s in with_semantics if s.semantics.display_kind == MetricDisplayKind.PERCENT
        }
        assert 100.0 in multipliers, 'no [0,1] ratio rendered as percent'
        assert 1.0 in multipliers, 'no official 0-100 scale sample'
        assert {'s', 'ms'} <= {s.semantics.display_unit for s in with_semantics}, 'a time unit is unpinned'
        assert any(
            s.semantics.display_kind == MetricDisplayKind.NUMBER and s.semantics.display_unit is None
            for s in with_semantics
        ), 'no unitless number'

        missing = [s for s in SAMPLES if s.value is None and s.semantics is not None]
        assert missing, 'no missing value sample with semantics'
        assert all(s.expected_primary == MISSING_PLACEHOLDER for s in missing)
        assert [s for s in SAMPLES if s.semantics is None and s.value is not None], 'no diagnostic fallback sample'

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
