import json
import os
import pandas as pd
from collections import defaultdict
from pydantic import BaseModel, Field, computed_field, field_serializer, field_validator, model_validator
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

from evalscope.api.metric.semantics import MetricRole, MetricSemantics
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
    name: str = 'default_metric'
    num: int = 0
    score: float = 0.0
    macro_score: float = 0.0
    semantic_id: Optional[str] = None
    """Persisted anchor into the semantics baseline table.

    Only the identifier is stored: the full contract is rebuilt on read by
    ``hydrate_report_semantics``, so a catalog correction applies to historical reports too and
    the file does not carry a copy of the semantics per metric.
    """

    categories: List[Category] = Field(default_factory=list)

    semantics: Optional[MetricSemantics] = Field(default=None, exclude=True)
    """Runtime-only resolved semantics, shared by every ``Category`` and ``Subset`` below.

    Excluded from serialization: ``semantic_id`` is the persisted form.
    """

    @model_validator(mode='after')
    def _compute_aggregates(self) -> Self:
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
        return self.semantics.role if self.semantics else None


class ReportKey:
    model_name = 'Model'
    dataset_name = 'Dataset'
    metric_name = 'Metric'
    category_name = 'Category'
    category_prefix = 'Cat.'
    subset_name = 'Subset'
    num = 'Num'
    score = 'Score'
    overall_score = 'OVERALL'


class Report(BaseModel):
    name: str = 'default_report'
    dataset_name: str = 'default_dataset'
    dataset_pretty_name: str = ''
    dataset_description: str = ''
    model_name: str = 'default_model'
    score: float = 0.0
    """Deprecated: kept for backward compatible readers.

    Retains its historical value (the first metric's score) and its non-null shape. Semantic
    consumers use :attr:`primary_metric_name` and ``Metric.semantics`` instead; nothing derives
    the primary metric from this number.
    """

    metrics: List[Metric] = Field(default_factory=list)
    analysis: str = 'N/A'
    # compare=False equivalent: excluded from model equality via model_config
    perf_metrics: Optional[Dict[str, Any]] = Field(default=None)
    primary_metric_name: Optional[str] = None
    """Final report metric name of the ``role=primary`` metric, ``None`` when there is none."""

    model_config = {'ignored_types': ()}

    @model_validator(mode='after')
    def _set_score(self) -> Self:
        if self.metrics:
            # Keep the historical number and shape of the deprecated `score` field.
            self.score = self.metrics[0].score
        declared = self._declared_primary_metric()
        if declared is not None:
            self.primary_metric_name = declared.name
        return self

    def _declared_primary_metric(self) -> Optional[Metric]:
        """Return the metric the benchmark *declared* as primary, if any.

        Kept separate from :meth:`_find_primary_metric` because ``primary_metric_name`` is not a
        display detail: ``hydrate_report_semantics`` reads it to decide which metric to promote to
        ``role=primary``. Writing an inferred choice there would make the inference decide the
        semantics, so only a declared metric is ever persisted under that name.

        A declared metric is one whose resolved semantics carry ``role=primary``; the resolver
        enforces at most one such metric per report, so scanning for it also covers the metric
        named by ``primary_metric_name``.
        """
        return next((m for m in self.metrics if m.role is MetricRole.PRIMARY), None)

    def _find_primary_metric(self) -> Optional[Metric]:
        """Return the metric that carries this report's conclusion.

        Resolution order, most to least authoritative:

        1. the metric declared primary, via :meth:`_declared_primary_metric`;
        2. the metric named by ``primary_metric_name``, which the benchmark declared through
           ``BenchmarkMeta.primary_metric`` and the report persisted, even if its resolved role
           disagrees;
        3. the first metric that is not a diagnostic;
        4. the first metric.

        Steps 3 and 4 are inferences, reported by :meth:`primary_metric_is_inferred`. They exist
        so every report presents a headline number: a report that shows nothing is less useful
        than one that shows its first real metric and says the choice was inferred. This is a read
        only: it never writes ``primary_metric_name``, and it never invents semantics -- an
        inferred metric is still formatted, coloured and compared strictly by its own contract.
        """
        declared = self._declared_primary_metric()
        if declared is not None:
            return declared
        if self.primary_metric_name:
            named = next((m for m in self.metrics if m.name == self.primary_metric_name), None)
            if named is not None:
                return named
        graded = next((m for m in self.metrics if m.role is not MetricRole.DIAGNOSTIC), None)
        return graded or (self.metrics[0] if self.metrics else None)

    def primary_metric_is_inferred(self) -> bool:
        """Whether the primary metric was inferred rather than declared.

        A consumer can mark an inferred headline as such, so "this benchmark says this is the
        conclusion" is never confused with "we picked something to show".
        """
        if not self.metrics:
            return False
        return self._declared_primary_metric() is None

    @computed_field
    @property
    def num(self) -> int:
        """Total sample count derived from the primary metric's subsets.

        Using a single metric avoids double-counting datasets that evaluate several metrics over
        the same sample set (e.g. multi_if reports 12 metrics over the same 6 samples). Falls
        back to the first metric while no primary metric is resolved.
        """
        metric = self._find_primary_metric()
        if metric is None:
            return 0
        return sum(s.num for c in metric.categories for s in c.subsets if not s.is_aggregate)

    @property
    def primary_metric(self) -> Optional[Metric]:
        """The metric carrying this report's conclusion, or ``None`` when it has no metric.

        Prefers the metric whose resolved semantics say ``role=primary``. When the benchmark
        declared none, a metric is *inferred* so the report still shows a headline number -- the
        first non-diagnostic metric, else the first one. There is no ``overall`` name convention.
        Call :meth:`primary_metric_is_inferred` to tell a declared conclusion from a chosen one;
        see :meth:`_find_primary_metric` for the full order.
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
        report = cls.model_validate(data)
        # Resolve the semantics of every metric on the single read path, so the API, the HTML
        # report, the CLI table and the DataFrame all see the same contract. Imported inside the
        # function to keep `report` importable without pulling in the semantics catalog.
        from evalscope.metrics.semantics import hydrate_report_semantics
        return hydrate_report_semantics(report)

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
