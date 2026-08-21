import json
import os
import pandas as pd
import re
from collections import defaultdict
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer, field_validator, model_validator
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

from evalscope.api.metric import JudgeSummary
from evalscope.api.metric.semantics import MetricIdentity, MetricKind, MetricSemantics
from evalscope.metrics import macro_mean, micro_mean
from evalscope.utils import get_logger
from evalscope.utils.argument_utils import get_secret_value

if TYPE_CHECKING:
    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.config import TaskConfig
    from evalscope.evaluation_versioning import BenchmarkEvaluationIdentity, ResolvedBenchmarkSpec

logger = get_logger()

ANALYSIS_PROMPT = """You are an expert AI model evaluator. Analyze the benchmark context and aggregated evaluation scores below and produce a concise, structured analysis report.

The report must contain exactly four sections with second-level Markdown headers (##):

## Overall Performance
Explain the benchmark's task goal and summarize the model's performance across its reported scores.

## Key Metrics Analysis
Break down scores by metric, category, and subset where available. If multiple metrics are present, categorize them into *Low*, *Medium*, and *High* performance tiers and present the breakdown in a Markdown table.

## Improvement Suggestions
Provide specific, actionable recommendations to address identified weaknesses or low-scoring areas.

## Conclusion
Offer a concise summary, including material limits of the reported evaluation scope.

Requirements:
- Output only the report content itself — no preamble, commentary, or closing remarks.
- Write the report in {language}.
- Keep the report focused and avoid unnecessary repetition.
- Assess only the provided evaluation scores; do not infer performance metrics or undocumented benchmark details.

```json
{analysis_context}
```
"""


class BenchmarkAnalysisContext(BaseModel):
    """Compact benchmark and score data supplied to report analysis."""

    benchmark: Dict[str, Any]
    resolved_benchmark: Dict[str, Any]
    results: Dict[str, Any]


def build_analysis_context(
    meta: 'BenchmarkMeta',
    spec: 'ResolvedBenchmarkSpec',
    identity: 'BenchmarkEvaluationIdentity',
    report: 'Report',
) -> BenchmarkAnalysisContext:
    """Build report-analysis input without documentation-only metadata or perf metrics."""
    overview, task_description = _description_sections(meta.description or '')
    return BenchmarkAnalysisContext(
        benchmark={
            'name': meta.name,
            'pretty_name': meta.pretty_name,
            'evaluation_version': identity.evaluation_version,
            'overview': overview,
            'task_description': task_description,
        },
        resolved_benchmark=spec.model_dump(mode='json'),
        results=_score_summary(report),
    )


def _description_sections(description: str) -> tuple[str, str]:
    sections: Dict[str, list[str]] = {'Overview': [], 'Task Description': []}
    current: Optional[str] = None
    for line in description.splitlines():
        heading = re.fullmatch(r'##\s+(.+?)\s*', line)
        if heading:
            current = heading.group(1) if heading.group(1) in sections else None
            continue
        if current is not None:
            sections[current].append(line)
    return '\n'.join(sections['Overview']).strip(), '\n'.join(sections['Task Description']).strip()


def _score_summary(report: 'Report') -> Dict[str, Any]:
    """Return only score aggregates relevant to a benchmark analysis."""
    return {
        'primary_metric_identity': (
            report.primary_metric_identity.model_dump(mode='json') if report.primary_metric_identity else None
        ),
        'metrics': [{
            'name': metric.name,
            'identity': metric.identity.model_dump(mode='json'),
            'num': metric.num,
            'score': metric.score,
            'macro_score': metric.macro_score,
            'categories': [{
                'name': list(category.name),
                'num': category.num,
                'score': category.score,
                'macro_score': category.macro_score,
                'subsets': [{
                    'name': subset.name,
                    'num': subset.num,
                    'score': subset.score,
                    'is_aggregate': subset.is_aggregate,
                } for subset in category.subsets],
            } for category in metric.categories],
        } for metric in report.metrics],
    }


def normalize_score(score: Union[float, dict, int], keep_num: int = 4) -> Union[float, dict]:
    """
    Normalize score.

    Args:
        score: input score, could be float or dict. e.g. 0.12345678 or {'acc': 0.12345678, 'f1': 0.12345678}
        keep_num: number of digits to keep.

    Returns:
        Union[float, dict]: normalized score. e.g. 0.1234 or {'acc': 0.1234, 'f1': 0.1234}
    """
    if isinstance(score, float):
        score = round(score, keep_num)
    elif isinstance(score, dict):
        score = {k: round(v, keep_num) for k, v in score.items()}
    elif isinstance(score, int):
        score = float(score)
    else:
        logger.warning(f'Unknown score type: {type(score)}')
    return score


class Subset(BaseModel):
    name: str = 'default_subset'
    score: float = 0.0
    num: int = 0
    is_aggregate: bool = False
    """True for derived/summary subsets (e.g. BFCL OVERALL, MULTI_TURN) that
    aggregate other real subsets. Excluded from num/score totals so they
    don't double-count, and hidden by the webview subset table."""

    @field_validator('score', mode='after')
    @classmethod
    def _normalize_score(cls, v: float) -> float:
        return normalize_score(v)


class Category(BaseModel):
    name: Tuple[str, ...] = Field(default_factory=tuple)
    num: int = 0
    score: float = 0.0
    macro_score: float = 0.0
    subsets: List[Subset] = Field(default_factory=list)

    @field_validator('name', mode='before')
    @classmethod
    def _coerce_name_to_tuple(cls, v) -> Tuple[str, ...]:
        if isinstance(v, str):
            return (v, )
        return tuple(v)

    @field_serializer('name')
    def _serialize_name(self, v: Tuple[str, ...]) -> List[str]:
        # Serialize as list for JSON compatibility (mirrors original asdict behaviour)
        return list(v)

    @model_validator(mode='after')
    def _compute_aggregates(self) -> Self:
        real = [s for s in self.subsets if not s.is_aggregate]
        self.num = sum(s.num for s in real)
        self.score = normalize_score(micro_mean(real)) if real else 0.0
        self.macro_score = normalize_score(macro_mean(real)) if real else 0.0
        return self


class Metric(BaseModel):
    model_config = ConfigDict(extra='forbid')

    identity: MetricIdentity
    legacy_name: Optional[str] = None
    num: int = 0
    score: float = 0.0
    macro_score: float = 0.0
    categories: List[Category] = Field(default_factory=list)

    semantics: MetricSemantics
    """Persisted display contract. Historical reports are resolved once during migration."""

    @model_validator(mode='before')
    @classmethod
    def _migrate_v1_shape(cls, data: Any) -> Any:
        from evalscope.metrics.semantics.migration import migrate_legacy_metric_payload
        return migrate_legacy_metric_payload(data)

    @model_validator(mode='after')
    def _compute_aggregates(self) -> Self:
        if not self.categories:
            return self
        # Categories whose subsets are all is_aggregate end up with num=0; skip them
        # so they don't drag down macro_mean.
        real = [c for c in self.categories if c.num > 0]
        self.num = sum(c.num for c in real)
        self.score = normalize_score(micro_mean(real)) if real else 0.0
        self.macro_score = normalize_score(macro_mean(real)) if real else 0.0
        return self

    @property
    def name(self) -> str:
        """Compatibility display key; v2 serialization stores ``identity`` instead."""
        return self.legacy_name or self.identity.key


class ReportKey:
    model_name = 'Model'
    dataset_name = 'Dataset'
    metric_name = 'Metric'
    category_name = 'Category'
    category_prefix = 'Cat.'
    subset_name = 'Subset'
    num = 'Num'
    score = 'Score'
    raw_score = 'Raw Score'
    display_score = 'Display Score'
    overall_score = 'OVERALL'


class ExecutionSubset(BaseModel):
    """Completion counts for one evaluated subset."""

    requested: int = 0
    succeeded: int = 0
    errored: int = 0


class ExecutionSummary(BaseModel):
    """Run completeness kept separate from aggregation-specific metric counts."""

    requested: int = 0
    succeeded: int = 0
    errored: int = 0
    incomplete: bool = False
    subsets: Dict[str, ExecutionSubset] = Field(default_factory=dict)


class Report(BaseModel):
    model_config = ConfigDict(extra='forbid')

    schema_version: int = 2
    name: str = 'default_report'
    dataset_name: str = 'default_dataset'
    dataset_pretty_name: str = ''
    dataset_description: str = ''
    model_name: str = 'default_model'
    metrics: List[Metric] = Field(default_factory=list)
    analysis: str = 'N/A'
    # compare=False equivalent: excluded from model equality via model_config
    perf_metrics: Optional[Dict[str, Any]] = Field(default=None)
    primary_metric_identity: Optional[MetricIdentity] = None
    judge_summary: Optional[JudgeSummary] = None
    """Run-level Native Judge coverage and failure summary, when this report used a judge."""
    execution_summary: Optional[ExecutionSummary] = None
    """Run completion status. Absent in reports written before completeness tracking."""

    @model_validator(mode='before')
    @classmethod
    def _migrate_v1_shape(cls, data: Any) -> Any:
        from evalscope.metrics.semantics.migration import migrate_legacy_report_payload
        return migrate_legacy_report_payload(data)

    @model_validator(mode='after')
    def _validate_v2(self) -> Self:
        if self.schema_version != 2:
            raise ValueError(f'unsupported report schema_version={self.schema_version}')
        if self.primary_metric_identity is not None:
            matches = [metric for metric in self.metrics if metric.identity == self.primary_metric_identity]
            if len(matches) != 1:
                raise ValueError('primary_metric_identity must match exactly one report metric')
            if matches[0].semantics.kind is MetricKind.DIAGNOSTIC:
                raise ValueError('primary_metric_identity must not reference a diagnostic metric')
        return self

    def _find_primary_metric(self) -> Optional[Metric]:
        """Return the metric identified by the persisted primary identity."""
        if self.primary_metric_identity is None:
            return None
        return next((metric for metric in self.metrics if metric.identity == self.primary_metric_identity), None)

    @computed_field
    @property
    def num(self) -> int:
        """Total sample count derived from one metric's subsets.

        Counting a single metric avoids double-counting datasets that evaluate several metrics over
        the same sample set (e.g. multi_if reports 12 metrics over the same 6 samples). Any one
        metric satisfies that, so a report whose primary metric could not be resolved still reports
        its real sample count instead of zero.
        """
        metric = self._find_primary_metric() or (self.metrics[0] if self.metrics else None)
        if metric is None:
            return 0
        return sum(s.num for c in metric.categories for s in c.subsets if not s.is_aggregate)

    @property
    def primary_metric(self) -> Optional[Metric]:
        """The metric carrying this report's conclusion, or ``None`` when it has no metric.

        A sole scored metric may be selected implicitly during generation; report reads use only
        the persisted identity and never fall back to list order.
        """
        return self._find_primary_metric()

    @property
    def score(self) -> Optional[float]:
        """Compatibility score derived from the primary or first available metric.

        ``None`` when the report carries no metric: a report that produced no usable score is
        not a report that scored zero. Report v2 serializes the structured metric list and
        primary identity instead of this convenience value.
        """
        metric = self._find_primary_metric() or (self.metrics[0] if self.metrics else None)
        return metric.score if metric is not None else None

    def to_dict(self) -> Dict[str, Any]:
        # model_dump includes computed_field 'num' automatically
        return self.model_dump()

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)

    def to_json(self, json_file: str):
        # ensure the directory exists
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        # write the report to a json file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict):
        # Pydantic handles nested model construction automatically via model_validate
        is_v2 = data.get('schema_version') == 2
        report = cls.model_validate(data, context={'migrating_v1': not is_v2})
        # Resolve the semantics of every metric on the single read path, so the API, the HTML
        # report, the CLI table and the DataFrame all see the same contract. Imported inside the
        # function to keep `report` importable without pulling in the semantics catalog.
        if is_v2:
            return report
        from evalscope.metrics.semantics import hydrate_report_semantics
        report = hydrate_report_semantics(report)
        if report.perf_metrics:
            from evalscope.metrics.semantics import attach_perf_semantics
            report.perf_metrics = attach_perf_semantics(report.perf_metrics)
        return report

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dataframe(
        self,
        flatten_metrics: bool = True,
        flatten_categories: bool = True,
        add_overall_metric: bool = False,
        include_aggregate: bool = False,
    ) -> pd.DataFrame:
        """
        Convert the report to a pandas DataFrame.
        Args:
            flatten_metrics (bool): Whether to flatten the metrics to a single row.
            flatten_categories (bool): Whether to flatten the categories to multiple rows.
            add_overall_metric (bool): Whether to add an overall metric row.
            include_aggregate (bool): Whether to emit derived/summary subsets
                (those with ``is_aggregate=True``). Off by default so they
                don't pollute per-subset tables in CLI/web views.
        Returns:
            pd.DataFrame: The report as a pandas DataFrame.
        """
        if not self.metrics:
            return pd.DataFrame()

        table = defaultdict(list)
        for metric in self.metrics:
            metric_count = 0
            for category in metric.categories:
                for subset in category.subsets:
                    if subset.is_aggregate and not include_aggregate:
                        continue
                    metric_count += 1
                    table[ReportKey.model_name].append(self.model_name)
                    table[ReportKey.dataset_name].append(self.dataset_name)
                    table[ReportKey.metric_name].append(metric.name)
                    table[ReportKey.category_name].append(category.name)
                    table[ReportKey.subset_name].append(subset.name)
                    table[ReportKey.num].append(subset.num)
                    table[ReportKey.score].append(subset.score)
            # add overall metric when there are multiple subsets
            if metric_count > 1 and add_overall_metric and (
                ReportKey.overall_score not in table[ReportKey.subset_name]
            ):
                table[ReportKey.model_name].append(self.model_name)
                table[ReportKey.dataset_name].append(self.dataset_name)
                table[ReportKey.metric_name].append(metric.name)
                table[ReportKey.category_name].append(('-', ))
                table[ReportKey.subset_name].append(ReportKey.overall_score)
                table[ReportKey.num].append(metric.num)
                table[ReportKey.score].append(metric.score)
            # NOTE: only flatten metrics if needed, use the first metric by default
            if not flatten_metrics:
                break
        df = pd.DataFrame.from_dict(table, orient='columns')
        if flatten_categories:
            df = self._flatten_categories(df)
        return df

    def _flatten_categories(self, df: pd.DataFrame):
        # expand categories to multiple rows
        df_categories = df.copy()
        # multi-level aggregation for categories
        max_depth = df_categories[ReportKey.category_name].apply(len).max()
        for level in range(max_depth):
            df_categories[f'{ReportKey.category_prefix}{level}'] = df_categories[
                ReportKey.category_name].apply(lambda x: x[level] if len(x) > level else None)

        df_categories.drop(columns=[ReportKey.category_name], inplace=True)
        return df_categories

    def generate_analysis(self, task_config: 'TaskConfig', analysis_context: 'BenchmarkAnalysisContext') -> str:
        """Generate score analysis from compact benchmark context and report aggregates."""
        from evalscope.constants import DEFAULT_LANGUAGE
        from evalscope.metrics import LLMJudge

        try:
            language = 'English' if DEFAULT_LANGUAGE == 'en' else 'Chinese'

            # Reuse the primary judge's transport configuration without making analysis part of
            # the scoring process.
            if task_config.judge.models:
                judge_model_config = task_config.judge.models[0].model_dump(exclude={'judge_id'}, exclude_none=True)
                judge_llm = LLMJudge(**get_secret_value(judge_model_config))
            else:
                judge_llm = LLMJudge(
                    api_key=get_secret_value(task_config.api_key),
                    api_url=task_config.api_url,
                    model_id=task_config.model,
                    eval_type=task_config.eval_type,
                )

            context_json = json.dumps(analysis_context.model_dump(mode='json'), ensure_ascii=False, indent=2)
            prompt = ANALYSIS_PROMPT.format(language=language, analysis_context=context_json)
            from evalscope.api.messages import ChatMessageUser

            response = judge_llm.generate([ChatMessageUser(content=prompt)]).completion
            if DEFAULT_LANGUAGE == 'en':
                disclaimer = f'> *Generated by {judge_llm.model_id}, for reference only.*'
            else:
                disclaimer = f'> *由 {judge_llm.model_id} 生成，仅供参考。*'
            response = f'{disclaimer}\n\n{response}'
        except Exception as e:
            logger.error(f'Error generating analysis: {e}')
            response = 'N/A'

        self.analysis = response
        return response
