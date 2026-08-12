import json
import os
import pandas as pd
from collections import defaultdict
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

from evalscope.api.metric.semantics import (
    MetricDirection,
    MetricDisplayKind,
    MetricIdentity,
    MetricRole,
    MetricSemantics,
)
from evalscope.metrics import macro_mean, micro_mean
from evalscope.utils import get_logger
from evalscope.utils.argument_utils import get_secret_value

if TYPE_CHECKING:
    from evalscope.config import TaskConfig

logger = get_logger()

ANALYSIS_PROMPT = """You are an expert AI model evaluator. Analyze the following JSON evaluation results and produce a concise, structured analysis report.

The report must contain exactly four sections with second-level Markdown headers (##):

## Overall Performance
Summarize the model's general performance across all evaluated benchmarks and metrics.

## Key Metrics Analysis
Break down individual metrics. If multiple metrics are present, categorize them into *Low*, *Medium*, and *High* performance tiers and present the breakdown in a Markdown table.

## Improvement Suggestions
Provide specific, actionable recommendations to address identified weaknesses or low-scoring areas.

## Conclusion
Offer a concise summary of the findings and an overall assessment.

Requirements:
- Output only the report content itself — no preamble, commentary, or closing remarks.
- Write the report in {language}.
- Keep the report focused and avoid unnecessary repetition.

```json
{report_str}
```
"""


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
        if not isinstance(data, dict) or 'identity' in data:
            return data
        migrated = dict(data)
        old_name = migrated.pop('name', 'legacy_metric')
        semantic_id = migrated.pop('semantic_id', None)
        migrated['identity'] = _migrate_legacy_report_identity(old_name)
        if 'semantics' not in migrated:
            migrated['semantics'] = _diagnostic_semantics(old_name, semantic_id)
        return migrated

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
    def role(self) -> Optional[MetricRole]:
        """Display tier of this metric, or ``None`` while the semantics are not resolved."""
        return self.semantics.role

    @property
    def name(self) -> str:
        """Compatibility display key; v2 serialization stores ``identity`` instead."""
        return self.legacy_name or self.identity.key


def _diagnostic_semantics(metric_name: str, semantic_id: Optional[str] = None) -> MetricSemantics:
    """Create a self-contained fallback without importing the resolver/catalog."""
    return MetricSemantics(
        semantic_id=semantic_id or 'diagnostic.unspecified',
        metric_name=metric_name,
        role=MetricRole.DIAGNOSTIC,
        direction=MetricDirection.NONE,
        display_kind=MetricDisplayKind.NUMBER,
        display_precision=4,
    )


def _migrate_legacy_report_identity(metric_name: str, benchmark_name: Optional[str] = None) -> MetricIdentity:
    """Migrate a known v1 name, or isolate an unknown legacy spelling as a diagnostic identity."""
    import re

    from evalscope.metrics.semantics.catalog import LEGACY_METRIC_MIGRATIONS
    from evalscope.metrics.semantics.identity import is_known_dynamic_legacy_name, migrate_legacy_identity

    if metric_name in LEGACY_METRIC_MIGRATIONS or is_known_dynamic_legacy_name(metric_name, benchmark_name):
        return migrate_legacy_identity(metric_name, 'identity', benchmark_name=benchmark_name)
    if re.fullmatch(r'[a-z][a-z0-9_]*', metric_name) and metric_name not in {'score', 'overall', 'total_score'}:
        return MetricIdentity(name=metric_name, aggregation='identity')
    return MetricIdentity(name='legacy_metric', aggregation='identity', dimensions={'original_name': metric_name})


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

    @model_validator(mode='before')
    @classmethod
    def _migrate_v1_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        migrated.pop('num', None)
        migrated.pop('metric_schema_version', None)
        metrics = migrated.get('metrics', [])
        has_v2_metrics = bool(metrics) and all(
            hasattr(metric, 'identity') or isinstance(metric, dict) and 'identity' in metric for metric in metrics
        )
        if migrated.get('schema_version') == 2 or has_v2_metrics:
            migrated.setdefault('schema_version', 2)
            return migrated

        dataset_name = migrated.get('dataset_name')
        primary_name = migrated.pop('primary_metric_name', None)
        migrated.pop('score', None)
        migrated['schema_version'] = 2
        migrated_metrics = []
        for metric_data in migrated.get('metrics', []):
            item = dict(metric_data)
            old_name = item.pop('name', 'legacy_metric')
            semantic_id = item.pop('semantic_id', None)
            identity = _migrate_legacy_report_identity(old_name, benchmark_name=dataset_name)
            item['identity'] = identity.model_dump()
            item['legacy_name'] = old_name
            item['semantics'] = _diagnostic_semantics(old_name, semantic_id).model_dump()
            migrated_metrics.append(item)
            if primary_name == old_name:
                migrated['primary_metric_identity'] = identity.model_dump()
        migrated['metrics'] = migrated_metrics
        return migrated

    @model_validator(mode='after')
    def _validate_v2(self, info: ValidationInfo) -> Self:
        if self.schema_version != 2:
            raise ValueError(f'unsupported report schema_version={self.schema_version}')
        declared_metrics = [metric for metric in self.metrics if metric.role is MetricRole.PRIMARY]
        if len(declared_metrics) > 1:
            raise ValueError('Report v2 must contain at most one metric with role=primary')
        declared = declared_metrics[0] if declared_metrics else None
        if self.primary_metric_identity is not None:
            matches = [metric for metric in self.metrics if metric.identity == self.primary_metric_identity]
            if len(matches) != 1:
                raise ValueError('primary_metric_identity must match exactly one report metric')
            if matches[0].role is not MetricRole.PRIMARY and not (info.context or {}).get('migrating_v1'):
                raise ValueError('primary_metric_identity must reference the metric with role=primary')
        elif declared is not None:
            self.primary_metric_identity = declared.identity
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
            from evalscope.metrics.semantics.perf import attach_perf_semantics
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

    def generate_analysis(self, task_config: 'TaskConfig') -> str:
        from evalscope.constants import DEFAULT_LANGUAGE
        from evalscope.metrics import LLMJudge

        try:
            language = 'English' if DEFAULT_LANGUAGE == 'en' else 'Chinese'

            # Use judge_model_args if configured; otherwise fall back to the task's own model settings
            if task_config.judge_model_args:
                judge_model_args = get_secret_value(task_config.judge_model_args)
                judge_llm = LLMJudge(**judge_model_args)
            else:
                judge_llm = LLMJudge(
                    api_key=get_secret_value(task_config.api_key),
                    api_url=task_config.api_url,
                    model_id=task_config.model,
                    eval_type=task_config.eval_type,
                )

            prompt = ANALYSIS_PROMPT.format(language=language, report_str=self.to_json_str())
            response = judge_llm.judge(prompt)
            if response.startswith('[ERROR]'):
                logger.warning(f'Analysis generation failed, skipping: {response}')
                response = 'N/A'
            else:
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
