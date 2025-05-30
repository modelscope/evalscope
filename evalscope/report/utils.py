import json
import os
import pandas as pd
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from evalscope.metrics import macro_mean, micro_mean
from evalscope.utils import normalize_score
from evalscope.utils.logger import get_logger

logger = get_logger()


@dataclass
class Subset:
    name: str = 'default_subset'
    score: float = 0.0
    num: int = 0

    def __post_init__(self):
        self.score = normalize_score(self.score)


@dataclass
class Category:
    name: tuple[str] = field(default_factory=tuple)
    num: int = 0
    score: float = 0.0
    macro_score: float = 0.0
    subsets: List[Subset] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.name, str):
            # ensure name is tuple format
            self.name = (self.name, )
        self.num = sum(subset.num for subset in self.subsets)
        self.score = normalize_score(micro_mean(self.subsets))
        self.macro_score = normalize_score(macro_mean(self.subsets))

    @classmethod
    def from_dict(cls, data: dict):
        subsets = [Subset(**subset) for subset in data.get('subsets', [])]
        return cls(name=data['name'], subsets=subsets)


@dataclass
class Metric:
    name: str = 'default_metric'
    num: int = 0
    score: float = 0.0
    macro_score: float = 0.0
    categories: List[Category] = field(default_factory=list)

    def __post_init__(self):
        self.num = sum(category.num for category in self.categories)
        self.score = normalize_score(micro_mean(self.categories))
        self.macro_score = normalize_score(macro_mean(self.categories))

    @classmethod
    def from_dict(cls, data: dict):
        categories = [Category.from_dict(category) for category in data.get('categories', [])]
        return cls(name=data['name'], categories=categories)


class ReportKey:
    model_name = 'Model'
    dataset_name = 'Dataset'
    metric_name = 'Metric'
    category_name = 'Category'
    category_prefix = 'Cat.'
    subset_name = 'Subset'
    num = 'Num'
    score = 'Score'


ANALYSIS_PROMPT = """根据给出的json格式的模型评测结果，输出分析报告，要求如下：
1. 报告分为 总体表现、关键指标分析、改进建议、结论 四部分
2. 若模型有多种指标，将其分为低分、中分、高分三个部分，并列出markdown表格
3. 只列出报告本身，不要有其他多余内容
4. 输出报告语言为{language}

```json
{report_str}
```
"""


@dataclass
class Report:
    name: str = 'default_report'
    dataset_name: str = 'default_dataset'
    dataset_pretty_name: str = ''
    dataset_description: str = ''
    model_name: str = 'default_model'
    score: float = 0.0
    metrics: List[Metric] = field(default_factory=list)
    analysis: str = 'N/A'

    def __post_init__(self):
        self.score = self.metrics[0].score  # NOTE: only use the first metric by default

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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
        metrics = [Metric.from_dict(metric) for metric in data.get('metrics', [])]
        return cls(
            name=data['name'],
            dataset_name=data['dataset_name'],
            dataset_pretty_name=data.get('dataset_pretty_name'),
            dataset_description=data.get('dataset_description'),
            score=data['score'],
            model_name=data['model_name'],
            metrics=metrics,
            analysis=data.get('analysis', 'N/A'),
        )

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dataframe(self, flatten_metrics: bool = True, flatten_categories: bool = True):
        table = defaultdict(list)
        for metric in self.metrics:
            for category in metric.categories:
                for subset in category.subsets:
                    table[ReportKey.model_name].append(self.model_name)
                    table[ReportKey.dataset_name].append(self.dataset_name)
                    table[ReportKey.metric_name].append(metric.name)
                    table[ReportKey.category_name].append(category.name)
                    table[ReportKey.subset_name].append(subset.name)
                    table[ReportKey.num].append(subset.num)
                    table[ReportKey.score].append(subset.score)
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
            df_categories[f'{ReportKey.category_prefix}{level}'] = df_categories[ReportKey.category_name].apply(
                lambda x: x[level] if len(x) > level else None)

        df_categories.drop(columns=[ReportKey.category_name], inplace=True)
        return df_categories

    def generate_analysis(self, judge_llm_config: dict) -> str:
        import locale

        from evalscope.metrics import LLMJudge

        try:
            # get the default locale
            lang, _ = locale.getlocale()

            if lang is None:
                language = '中文'
            else:
                language = 'en' if lang.startswith('en') else '中文'

            prompt = ANALYSIS_PROMPT.format(language=language, report_str=self.to_json_str())
            judge_llm = LLMJudge(**judge_llm_config)
            response = judge_llm(prompt)
        except Exception as e:
            logger.error(f'Error generating analysis: {e}')
            response = 'N/A'

        self.analysis = response
        return response
