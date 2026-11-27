"""Persisted, benchmark-independent facts and sample diagnostics."""

import re
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from evalscope.api.model.generate_config import GenerateConfig

from .config import TraceAnalysisConfig
from .taxonomy import ISSUE_NAMES, TAXONOMY_VERSION
from .trace import Attribution, Finding, FindingCategory


class DiagnosticData(BaseModel):
    """Persisted outputs always include defaults; Web contracts describe that complete shape."""

    model_config = ConfigDict(extra='forbid', json_schema_serialization_defaults_required=True)


class SampleKey(DiagnosticData):
    """A persisted sample identity within one model and benchmark."""

    subset: str
    index: int


class AnalysisModelConfig(BaseModel):
    """Explicit diagnostic model; never silently reuse scoring credentials or enable paid analysis."""

    model_config = ConfigDict(extra='forbid')
    model_id: str = Field(min_length=1)
    api_url: Optional[str] = None
    api_key: Optional[SecretStr] = None
    eval_type: str = 'openai_api'
    generation_config: GenerateConfig = Field(default_factory=lambda: GenerateConfig(temperature=0, max_tokens=2048))


class AnalysisFactsConfig(BaseModel):
    """Settings shared by both levels, without model or sampling options."""

    model_config = ConfigDict(extra='forbid')

    language: Literal['auto', 'zh', 'en'] = 'auto'
    trace: TraceAnalysisConfig = Field(default_factory=TraceAnalysisConfig)
    weak_subset_gap: float = Field(default=0.1, gt=0)
    weak_subset_min_samples: int = Field(default=20, ge=1)
    max_examples: int = Field(default=5, ge=1, le=20)
    warning_rate: float = Field(default=0.1, ge=0, le=1)


class AnalysisReportConfig(AnalysisFactsConfig):
    """Facts make no model calls; diagnose requires an explicit model for sample hypotheses and a summary."""

    level: Literal['facts', 'diagnose'] = 'facts'
    model: Optional[AnalysisModelConfig] = None
    max_diagnosed_samples: Optional[int] = Field(default=50, ge=1, strict=True)
    scope: Literal['failed', 'unscored', 'risk', 'process', 'all'] = 'failed'
    samples: Optional[List[SampleKey]] = Field(default=None, min_length=1)
    concurrency: int = Field(default=4, ge=1, le=32)
    seed: int = 42
    include_trace: bool = True
    max_input_chars: int = Field(default=24000, ge=2000, le=100000)

    def facts_settings(self) -> AnalysisFactsConfig:
        """Extract only the settings that determine facts and rule observations."""
        return AnalysisFactsConfig.model_validate(self.model_dump(include=set(AnalysisFactsConfig.model_fields)))

    @classmethod
    def from_source(cls, source: str) -> 'AnalysisReportConfig':
        """Read a JSON object or JSON file; unknown options are rejected."""
        from pathlib import Path

        text = source if source.lstrip().startswith('{') else Path(source).read_text()
        return cls.model_validate_json(text)


class Fact(DiagnosticData):
    """An observation with an explicit denominator and computation definition."""

    id: str
    kind: Literal['generic', 'generation', 'mcq', 'repeats', 'agent', 'judge', 'perf']
    severity: Literal['info', 'warn', 'critical'] = 'info'
    title: str
    value: Optional[float] = None
    unit: str = ''
    definition: str
    numerator: Optional[int] = None
    denominator: Optional[int] = None
    samples: List[SampleKey] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)


class EvidenceReference(DiagnosticData):
    """Offsets are zero-based Python Unicode character offsets, end-exclusive, in the supplied source."""

    source_id: str
    start: Optional[int] = Field(default=None, ge=0)
    end: Optional[int] = Field(default=None, gt=0)
    quote: str = Field(min_length=1, max_length=2000)


class DiagnosticSource(DiagnosticData):
    """Redacted exact text supplied to the model, with an optional original message/step identity."""

    id: str
    text: str
    message_index: Optional[int] = None
    step: Optional[int] = None
    excerpted: bool = False
    original_start: int = 0


class LLMDiagnosis(DiagnosticData):
    """A model hypothesis; valid citations do not establish semantic correctness."""

    failure_type: str = Field(json_schema_extra={'enum': list(ISSUE_NAMES)})
    secondary_types: List[Annotated[str, Field(json_schema_extra={'enum': list(ISSUE_NAMES)})]] = Field(
        default_factory=list, max_length=3
    )
    attribution: Attribution
    root_cause_step: Optional[int] = Field(default=None, ge=0, strict=True)
    explanation: str = Field(min_length=1, max_length=1200)
    evidence: List[EvidenceReference] = Field(default_factory=list, max_length=3)
    confidence: Literal['low', 'medium', 'high']
    gold_suspect: bool = False

    @field_validator('failure_type', 'secondary_types')
    @classmethod
    def known_types(cls, value: Any) -> Any:
        """Unknown taxonomy values are parse failures, never reusable successful responses."""
        values = value if isinstance(value, list) else [value]
        if any(item not in ISSUE_NAMES for item in values):
            raise ValueError('Unknown diagnostic type')
        return value


class NarrativeSampleReference(DiagnosticData):
    """A summary citation resolved by the analyzer to one inspected sample and optional evidence step."""

    sample: SampleKey
    step: Optional[int] = Field(default=None, ge=0)


class NarrativeItem(DiagnosticData):
    """Every sentence has resolvable fact/sample references; it remains model-generated text."""

    text: str = Field(min_length=1, max_length=2000)
    refs: List[str] = Field(min_length=1, max_length=8)

    @field_validator('text')
    @classmethod
    def separate_references(cls, text: str) -> str:
        """Render references once through the validated ref list, without leaking internal IDs in prose."""
        text = re.sub(r'(?:,\s*)?\b[FPS]-[\w-]+\b', '', text)
        return re.sub(r'\s*(?:\[\s*\]|\(\s*\)|（\s*）|【\s*】)', '', text).strip()


class Narrative(DiagnosticData):
    """Grounded summary and suggestions, without an unreferenced free-form summary field."""

    findings: List[NarrativeItem] = Field(default_factory=list, max_length=3)
    recommendations: List[NarrativeItem] = Field(default_factory=list, max_length=3)
    limitations: List[str] = Field(default_factory=list, max_length=5)


class SampleDiagnosis(DiagnosticData):
    """Official outcome and observed findings; a finding is not a proved root cause."""

    key: SampleKey
    task_success: Optional[bool] = None
    safety_violation: Optional[bool] = None
    score_status: str
    stop_reason: Optional[str] = None
    needs_attention: bool = False
    findings: List[Finding] = Field(default_factory=list)
    observed_types: List[str] = Field(default_factory=list)
    recorded_error: Optional[str] = None
    llm: Optional[LLMDiagnosis] = None
    llm_status: Literal['not_selected', 'ok', 'parse_error', 'transport_error', 'invalid_evidence', 'unexplained'] = (
        'not_selected'
    )
    sources: List[DiagnosticSource] = Field(default_factory=list)
    context_omitted: bool = False
    invalid_evidence: int = 0
    evidence_errors: List[str] = Field(default_factory=list)

    @property
    def is_unscored(self) -> bool:
        return self.score_status not in ('success', 'fallback', 'degraded')

    @property
    def has_risk(self) -> bool:
        return self.safety_violation is True or any(
            finding.category in (FindingCategory.RISKY_COMMAND, FindingCategory.ATTACK_GOAL_ACTION)
            for finding in self.findings
        )

    @property
    def issue_types(self) -> List[str]:
        if self.observed_types:
            return self.observed_types
        return ['unexplained'] if self.task_success is False or self.is_unscored else []


class FailurePattern(DiagnosticData):
    """Observed issue types among failed or unscored samples; groups may overlap."""

    failure_type: str
    failed: int = 0
    unscored: int = 0
    tools: Dict[str, int] = Field(default_factory=dict)
    examples: List[SampleKey] = Field(default_factory=list)


class DiagnosisCoverage(DiagnosticData):
    """Do not equate unavailable outcomes or absent facts with success or zero events."""

    samples: int = 0
    task_success_known: int = 0
    task_failed: int = 0
    outcome_unknown: int = 0
    excluded: int = 0
    errored_without_record: int = 0
    unreadable: int = 0
    unexplained_failures: int = 0
    risk_samples: int = 0
    proposals_without_rule_clues: int = 0
    skipped_facts: Dict[str, str] = Field(default_factory=dict)
    analysis_errors: Dict[str, int] = Field(default_factory=dict)
    selection_scope: Optional[Literal['failed', 'unscored', 'risk', 'process', 'all', 'specified']] = None
    diagnosis_population: int = 0
    selected: int = 0
    diagnosed_ok: int = 0
    diagnosis_statuses: Dict[str, int] = Field(default_factory=dict)
    citations_total: int = 0
    citations_valid: int = 0


class DiagnosisCost(DiagnosticData):
    """Analysis usage only, separate from evaluation and scoring usage."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_calls: int = 0
    estimated_calls: int = 0
    estimated_input_tokens: int = 0
    usage_missing_calls: int = 0


class DiagnosisReport(DiagnosticData):
    """One analysis of a persisted (run, model, benchmark), with cache provenance."""

    schema_version: int = 7
    analyzer_version: str
    taxonomy_version: str = TAXONOMY_VERSION
    run_dir: str
    model: str
    benchmark: str
    level: Literal['facts', 'diagnose'] = 'facts'
    diagnosis_model: Optional[str] = None
    self_evaluation: bool = False
    narrative: Optional[Narrative] = None
    narrative_status: str = 'not_requested'
    narrative_samples: Dict[str, NarrativeSampleReference] = Field(default_factory=dict)
    generated_at: str
    config_fingerprint: str
    data_fingerprint: str
    samples_sha256: str = ''
    coverage: DiagnosisCoverage
    cost: DiagnosisCost = Field(default_factory=DiagnosisCost)
    facts: List[Fact] = Field(default_factory=list)
    patterns: List[FailurePattern] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
