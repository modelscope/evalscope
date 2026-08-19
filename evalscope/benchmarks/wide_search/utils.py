# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pandas as pd
import re
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from evalscope.api.metric import AggScore, SampleScore

METRIC_NAMES = (
    'success_rate',
    'row_precision',
    'row_recall',
    'row_f1',
    'item_precision',
    'item_recall',
    'item_f1',
)

PRIMARY_KEY_PREPROCESS_PROMPT = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.


The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
    "mapping": {{
        "origin_str1": "transform_str1",
        "origin_str2": "transform_str2"
    }}
}}
```
"""  # noqa: E501

EVAL_COLUMN_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, provide the final scoring results. Each pair should have a score. Reply with a single JSON object and no prose, as shown below:
```json
{{
    "idx_xxx": score,
    "idx_yyy": score,
    ...
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======

Now start scoring. Please make sure to analyze each item step by step before providing the final scoring results.

"""  # noqa: E501


def norm_column(column: str) -> str:
    return str(column).strip().lower().replace(' ', '')


def extract_markdown_table(response: str) -> Optional[pd.DataFrame]:
    markdown_matches = re.findall(r'```markdown(.*?)```', response, re.DOTALL)
    if not markdown_matches:
        pipe_positions = [match.start() for match in re.finditer(r'\|', response)]
        if len(pipe_positions) >= 4:
            first_pipe = pipe_positions[0]
            last_pipe = pipe_positions[-1]
            start = response.rfind('\n', 0, first_pipe)
            start = 0 if start == -1 else start
            end = response.find('\n', last_pipe)
            end = len(response) if end == -1 else end
            table_candidate = response[start:end]
            markdown_matches = re.findall(r'((?:\|.*\n?)+)', table_candidate)
    if not markdown_matches:
        return None

    markdown = markdown_matches[0].strip()
    lines = markdown.split('\n')
    lines[0] = lines[0].replace(' ', '').lower()
    normalized_lines = []
    for line in (line.strip() for line in lines):
        if set(line).issubset(set('|- :')) or '|' not in line:
            continue
        normalized_lines.append('|'.join(part.strip() for part in line.split('|')))
    markdown = '\n'.join(normalized_lines)
    response_df = pd.read_csv(StringIO(markdown), sep='|')
    return response_df.loc[:, ~response_df.columns.str.startswith('Unnamed')]


def extract_number(content: str) -> str:
    numbers = re.findall(r'[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?', str(content).replace(',', ''))
    return numbers[0] if numbers else 'NULL'


def norm_str(content: str) -> str:
    return str(content).lower().strip().replace(' ', '').replace('*', '')


def norm_date(content: str) -> str:
    import dateparser

    normalized_date = dateparser.parse(content, settings={'PREFER_DAY_OF_MONTH': 'first'})
    return content if normalized_date is None else normalized_date.strftime('%Y-%m-%d')


def exact_match(response: str, target: str) -> float:
    return float(response.lower() == target.lower())


def url_match(response: str, target: str) -> float:
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    response_urls = [urlparse(url).netloc for url in pattern.findall(response)]
    target_urls = [urlparse(url).netloc for url in pattern.findall(target)]
    return float(set(response_urls) == set(target_urls))


def in_match(response: str, target: str) -> float:
    return float(response in target)


def number_near(response: str, target: str, criterion: float) -> float:
    response_num = _parse_number(response)
    target_num = _parse_number(target)
    if response_num is None or target_num is None:
        return float(response_num is None and target_num is None and response == target)
    return float(abs(response_num - target_num) <= abs(target_num) * criterion)


def _parse_number(content: str) -> Optional[float]:
    try:
        if '%' in content:
            return float(content.replace('%', '')) / 100.0
        return float(content)
    except (TypeError, ValueError):
        return None


def date_near(response: str, target: str) -> float:
    import dateparser

    try:
        response_date = dateparser.parse(response, settings={'PREFER_DAY_OF_MONTH': 'first'})
    except Exception:
        response_date = None
    try:
        target_date = dateparser.parse(target, settings={'PREFER_DAY_OF_MONTH': 'first'})
    except Exception:
        target_date = None
    if response_date is None or target_date is None:
        return float(response_date is None and target_date is None)
    return float(abs((response_date - target_date).days) <= 31)


PREPROCESSORS: Dict[str, Callable[[str], str]] = {
    'extract_number': extract_number,
    'norm_str': norm_str,
    'norm_date': norm_date,
}


@dataclass
class WideSearchSession:
    """Pure, official WideSearch table preparation and reduction.

    The adapter turns the LLM-dependent alignment and column scoring stages into judge cases;
    this class deliberately contains no model callback and no parsing of judge text.
    """

    answer_df: pd.DataFrame
    response_df: Optional[pd.DataFrame]
    evaluation: Dict[str, Any]
    error: Optional[str] = None

    @classmethod
    def create(cls, prediction: str, gold_csv: str, evaluation: Dict[str, Any]) -> 'WideSearchSession':
        try:
            required = list(evaluation['required'])
            answer_df = pd.read_csv(StringIO(gold_csv))
            answer_df.columns = [norm_column(column) for column in answer_df.columns]
            answer_df = answer_df[required]
            response_df = extract_markdown_table(prediction)
            if response_df is None:
                return cls(answer_df=answer_df, response_df=None, evaluation=evaluation, error='response_df is None')
            response_df.columns = [norm_column(column) for column in response_df.columns]
            return cls(answer_df=answer_df, response_df=response_df, evaluation=evaluation)
        except Exception as error:
            return cls(
                answer_df=pd.DataFrame(),
                response_df=None,
                evaluation=evaluation,
                error=f'{type(error).__name__}: {error}'
            )

    @property
    def required_columns(self) -> List[str]:
        return list(self.evaluation['required'])

    @property
    def unique_columns(self) -> List[str]:
        return list(self.evaluation['unique_columns'])

    def needs_column_alignment(self) -> bool:
        return self.response_df is not None and set(self.required_columns) != set(self.response_df.columns)

    def frames(
        self,
        column_map: Optional[Dict[str, str]] = None,
        primary_key_maps: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {'stage': 'align'}
        if self.error or self.response_df is None:
            diagnostics['error'] = self.error or 'response_df is None'
            return None, None, diagnostics
        answer_df = self.answer_df.copy()
        response_df = self.response_df.copy()
        if column_map:
            response_df.rename(columns=_unique_target_map(column_map), inplace=True)
        if set(self.required_columns) != set(response_df.columns):
            diagnostics['error'] = 'required columns do not match response columns'
            diagnostics['response_columns'] = response_df.columns.tolist()
            return None, None, diagnostics
        for column in self.required_columns:
            answer_df[column] = answer_df[column].astype(str)
            response_df[column] = response_df[column].astype(str)
        response_df.drop_duplicates(subset=self.unique_columns, inplace=True)
        answer_df.drop_duplicates(subset=self.unique_columns, inplace=True)
        for column, value_map in (primary_key_maps or {}).items():
            response_df[column] = response_df[column].apply(lambda value: value_map.get(value, value))
        for column, pipeline in self.evaluation['eval_pipeline'].items():
            for preprocess_name in pipeline.get('preprocess', []):
                preprocess = PREPROCESSORS[preprocess_name]
                response_df[column] = response_df[column].apply(preprocess)
                answer_df[column] = answer_df[column].apply(preprocess)
        return answer_df, response_df, diagnostics

    def inner_frame(
        self,
        column_map: Optional[Dict[str, str]] = None,
        primary_key_maps: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        answer_df, response_df, diagnostics = self.frames(column_map, primary_key_maps)
        if answer_df is None or response_df is None:
            return None, diagnostics
        inner_df = pd.merge(
            answer_df,
            response_df,
            on=self.unique_columns,
            how='inner',
            suffixes=('_query', '_response'),
        )
        diagnostics.update({
            'gold_rows': len(answer_df),
            'prediction_rows': len(response_df),
            'matched_rows': len(inner_df),
        })
        return inner_df, diagnostics

    def score(
        self,
        column_scores: Dict[str, List[float]],
        column_map: Optional[Dict[str, str]] = None,
        primary_key_maps: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> tuple[Dict[str, float], Dict[str, Any]]:
        inner_df, diagnostics = self.inner_frame(column_map, primary_key_maps)
        if inner_df is None or inner_df.empty:
            return {name: 0.0 for name in METRIC_NAMES}, diagnostics
        inner_scores = pd.DataFrame(index=inner_df.index)
        for column in self.required_columns:
            if column in self.unique_columns:
                inner_scores[f'{column}_exact_match'] = 1.0
                continue
            pipeline = self.evaluation['eval_pipeline'][column]
            criterion = pipeline.get('criterion')
            for metric_name in pipeline.get('metric', []):
                key = f'{column}_{metric_name}'
                if metric_name == 'llm_judge':
                    values = column_scores[key]
                    if len(values) != len(inner_df):
                        raise ValueError(
                            f'WideSearch judge case {key} scored {len(values)} rows, expected {len(inner_df)}.'
                        )
                else:
                    values = [
                        _metric_call(response, target, criterion, metric_name)
                        for response, target in zip(inner_df[f'{column}_response'], inner_df[f'{column}_query'])
                    ]
                inner_scores[key] = values
        row_scores = inner_scores.min(axis=1)
        true_positive_rows = float(row_scores.sum())
        true_positive_items = float(inner_scores.sum().sum())
        _, response_df, _ = self.frames(column_map, primary_key_maps)
        prediction_rows = len(response_df) if response_df is not None else 0
        gold_rows = len(self.answer_df)
        row_precision = true_positive_rows / prediction_rows if prediction_rows else 0.0
        row_recall = true_positive_rows / gold_rows if gold_rows else 0.0
        item_precision = true_positive_items / (
            prediction_rows * len(self.required_columns)
        ) if prediction_rows else 0.0
        item_recall = true_positive_items / (gold_rows * len(self.required_columns)) if gold_rows else 0.0
        row_f1 = _f1(row_precision, row_recall)
        item_f1 = _f1(item_precision, item_recall)
        return {
            'success_rate': float(
                row_precision == row_recall == row_f1 == 1.0 and item_precision == item_recall == item_f1 == 1.0
            ),
            'row_precision': row_precision,
            'row_recall': row_recall,
            'row_f1': row_f1,
            'item_precision': item_precision,
            'item_recall': item_recall,
            'item_f1': item_f1,
        }, diagnostics


def _metric_call(response: str, target: str, criterion: Any, metric_name: str) -> float:
    if metric_name == 'exact_match':
        return exact_match(response, target)
    if metric_name == 'url_match':
        return url_match(response, target)
    if metric_name == 'in_match':
        return in_match(response, target)
    if metric_name == 'number_near':
        return number_near(response, target, float(criterion))
    if metric_name == 'date_near':
        return date_near(response, target)
    raise ValueError(f'Unsupported WideSearch metric: {metric_name}')


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 1e-9 else 0.0


def _unique_target_map(column_map: Dict[str, str]) -> Dict[str, str]:
    unique_map: Dict[str, str] = {}
    seen_targets = set()
    for source, target in column_map.items():
        if target in seen_targets:
            continue
        unique_map[source] = target
        seen_targets.add(target)
    return unique_map


def aggregate_official_scores(sample_scores: List[SampleScore]) -> List[AggScore]:
    """Aggregate official Avg@N, Pass@N and Max@N for all/en/zh scopes."""
    results: List[AggScore] = []
    scopes = {
        'all': sample_scores,
        'en': [score for score in sample_scores if (score.sample_metadata or {}).get('language') == 'en'],
        'zh': [score for score in sample_scores if (score.sample_metadata or {}).get('language') == 'zh'],
    }
    for scope, scoped_scores in scopes.items():
        if not scoped_scores:
            continue
        grouped: Dict[Any, List[SampleScore]] = defaultdict(list)
        for score in scoped_scores:
            group_id = score.group_id if score.group_id is not None else score.sample_id
            grouped[group_id].append(score)
        repeat_counts = {len(group) for group in grouped.values()}
        if len(repeat_counts) != 1:
            raise ValueError(f'WideSearch requires the same number of trials per task, got {sorted(repeat_counts)}.')
        repeats = repeat_counts.pop()
        sample_ids = [score.sample_id for score in scoped_scores]
        for metric_name in METRIC_NAMES:
            if metric_name.startswith('row_'):
                canonical_name = metric_name[4:]
                target = 'row'
            elif metric_name.startswith('item_'):
                canonical_name = metric_name[5:]
                target = 'item'
            else:
                canonical_name = metric_name
                target = None
            dimensions = {'scope': scope, 'k': repeats}
            if target is not None:
                dimensions['target'] = target
            all_values = [float(score.score.value[metric_name]) for score in scoped_scores]
            results.append(
                AggScore(
                    metric_name=canonical_name,
                    score=sum(all_values) / len(all_values),
                    aggregation='mean',
                    dimensions=dimensions,
                    num=len(all_values),
                    ids=sample_ids,
                )
            )
            group_maxima = [max(float(score.score.value[metric_name]) for score in group) for group in grouped.values()]
            aggregate_name = 'pass' if metric_name == 'success_rate' else 'max'
            results.append(
                AggScore(
                    metric_name=canonical_name,
                    score=sum(group_maxima) / len(group_maxima),
                    aggregation='pass_at_k' if aggregate_name == 'pass' else 'max',
                    dimensions=dimensions,
                    num=len(group_maxima),
                    ids=list(grouped.keys()),
                )
            )
    return results
