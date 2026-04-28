import json
import os
import pandas as pd
from collections import defaultdict
from pydantic import BaseModel, Field, computed_field, field_serializer, field_validator, model_validator
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

from evalscope.metrics import macro_mean, micro_mean
from evalscope.utils import get_logger

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
        self.num = sum(subset.num for subset in self.subsets)
        self.score = normalize_score(micro_mean(self.subsets))
        self.macro_score = normalize_score(macro_mean(self.subsets))
        return self


class Metric(BaseModel):
    name: str = 'default_metric'
    num: int = 0
    score: float = 0.0
    macro_score: float = 0.0
    categories: List[Category] = Field(default_factory=list)

    @model_validator(mode='after')
    def _compute_aggregates(self) -> Self:
        self.num = sum(category.num for category in self.categories)
        self.score = normalize_score(micro_mean(self.categories))
        self.macro_score = normalize_score(macro_mean(self.categories))
        return self


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
    metrics: List[Metric] = Field(default_factory=list)
    analysis: str = 'N/A'
    # compare=False equivalent: excluded from model equality via model_config
    perf_metrics: Optional[Dict[str, Any]] = Field(default=None)

    model_config = {'ignored_types': ()}

    @model_validator(mode='after')
    def _set_score(self) -> Self:
        if self.metrics:
            self.score = self.metrics[0].score  # NOTE: only use the first metric by default
        return self

    @computed_field
    @property
    def num(self) -> int:
        """Total sample count derived from the first metric's subsets.

        Using the first metric avoids double-counting datasets that have
        multiple metrics over the same sample set (e.g. multi_if has 12
        metrics all evaluated on the same 6 samples).
        """
        first = self.metrics[0] if self.metrics else None
        if first is None:
            return 0
        return sum(s.num for c in first.categories for s in c.subsets)

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
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dataframe(
        self,
        flatten_metrics: bool = True,
        flatten_categories: bool = True,
        add_overall_metric: bool = False
    ) -> pd.DataFrame:
        """
        Convert the report to a pandas DataFrame.
        Args:
            flatten_metrics (bool): Whether to flatten the metrics to a single row.
            flatten_categories (bool): Whether to flatten the categories to multiple rows.
            add_overall_metric (bool): Whether to add an overall metric row.
        Returns:
            pd.DataFrame: The report as a pandas DataFrame.
        """
        table = defaultdict(list)
        for metric in self.metrics:
            metric_count = 0
            for category in metric.categories:
                for subset in category.subsets:
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
                judge_llm = LLMJudge(**task_config.judge_model_args)
            else:
                judge_llm = LLMJudge(
                    api_key=task_config.api_key,
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
