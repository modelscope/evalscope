# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import pandas as pd
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from evalscope.agent.environments.local import LocalAgentEnvironment
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
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```
"""  # noqa: E501

EVAL_COLUMN_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
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


@dataclass
class EvaluationResult:
    values: Dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in METRIC_NAMES})
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class TemporaryLocalAgentEnvironment(LocalAgentEnvironment):
    """Local agent environment with a per-sample temporary working directory."""

    def __init__(self, sample_id: Any) -> None:
        safe_id = ''.join(char if str(char).isalnum() else '-' for char in str(sample_id))[:64]
        self._temporary_directory = tempfile.TemporaryDirectory(prefix=f'evalscope-wide-search-{safe_id}-')
        super().__init__(working_dir=self._temporary_directory.name)

    @property
    def working_dir(self) -> Path:
        return Path(self._temporary_directory.name)

    async def close(self) -> None:
        await super().close()
        self._temporary_directory.cleanup()


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


def parse_markdown_json(completion: str) -> Optional[Dict[str, Any]]:
    matches = re.findall(r'```json\s*(\{.*?\})\s*```', completion, re.DOTALL)
    if not matches:
        return None
    try:
        parsed = json.loads(matches[-1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


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


class WideSearchScorer:
    """Official WideSearch table aligner and hybrid scorer."""

    def __init__(self, judge: Callable[[str], str]) -> None:
        self.judge = judge

    def evaluate(self, prediction: str, gold_csv: str, evaluation: Dict[str, Any]) -> EvaluationResult:
        diagnostics: Dict[str, Any] = {'stage': 'parse'}
        try:
            required_columns = list(evaluation['required'])
            unique_columns = list(evaluation['unique_columns'])
            answer_df = pd.read_csv(StringIO(gold_csv))
            answer_df.columns = [norm_column(column) for column in answer_df.columns]
            answer_df = answer_df[required_columns]
            response_df = extract_markdown_table(prediction)
            if response_df is None:
                diagnostics['error'] = 'response_df is None'
                return EvaluationResult(diagnostics=diagnostics)

            response_df.columns = [norm_column(column) for column in response_df.columns]
            if set(required_columns) != set(response_df.columns):
                column_map, judge_response = self._map_values(response_df.columns.tolist(), required_columns)
                diagnostics['column_map'] = column_map
                diagnostics['column_map_judge'] = judge_response
                response_df.rename(columns=column_map, inplace=True)
            if set(required_columns) != set(response_df.columns):
                diagnostics['error'] = 'required columns do not match response columns'
                diagnostics['response_columns'] = response_df.columns.tolist()
                return EvaluationResult(diagnostics=diagnostics)

            diagnostics['stage'] = 'align'
            for column in required_columns:
                try:
                    answer_type = answer_df[column].dtype
                    response_type = response_df[column].dtype
                except Exception:
                    answer_type = None
                    response_type = None
                if (response_type == float and answer_type == int) or (response_type == int and answer_type == float):
                    if response_type == int:
                        response_df[column] = response_df[column].astype(float)
                    elif answer_type == int:
                        answer_df[column] = answer_df[column].astype(float)
                answer_df[column] = answer_df[column].astype(str)
                response_df[column] = response_df[column].astype(str)

            response_df.drop_duplicates(subset=unique_columns, inplace=True)
            answer_df.drop_duplicates(subset=unique_columns, inplace=True)
            diagnostics['primary_key_maps'] = {}
            diagnostics['primary_key_judges'] = {}
            for column in unique_columns:
                pipeline = evaluation['eval_pipeline'].get(column)
                if pipeline is None:
                    continue
                metrics = pipeline.get('metric', [])
                if 'llm_judge' in metrics or 'exact_match' in metrics:
                    value_map, judge_response = self._map_values(
                        response_df[column].tolist(), answer_df[column].tolist()
                    )
                    diagnostics['primary_key_maps'][column] = value_map
                    diagnostics['primary_key_judges'][column] = judge_response
                    response_df[column] = response_df[column].apply(lambda value: value_map.get(value, value))

            for column, pipeline in evaluation['eval_pipeline'].items():
                for preprocess_name in pipeline.get('preprocess', []):
                    preprocess = PREPROCESSORS[preprocess_name]
                    response_df[column] = response_df[column].apply(preprocess)
                    answer_df[column] = answer_df[column].apply(preprocess)

            inner_df = pd.merge(
                answer_df,
                response_df,
                on=unique_columns,
                how='inner',
                suffixes=('_query', '_response'),
            )
            diagnostics.update({
                'gold_rows': len(answer_df),
                'prediction_rows': len(response_df),
                'matched_rows': len(inner_df),
            })

            diagnostics['stage'] = 'score'
            inner_scores = pd.DataFrame(index=inner_df.index)
            diagnostics['column_judges'] = {}
            for column in required_columns:
                if column in unique_columns:
                    inner_scores[f'{column}_exact_match'] = 1.0
                    continue
                pipeline = evaluation['eval_pipeline'][column]
                criterion = pipeline.get('criterion')
                for metric_name in pipeline.get('metric', []):
                    if metric_name == 'llm_judge':
                        values, judge_response = self._judge_column(
                            inner_df[f'{column}_response'].tolist(),
                            inner_df[f'{column}_query'].tolist(),
                            criterion,
                        )
                        diagnostics['column_judges'][column] = judge_response
                    else:
                        values = [
                            self._metric_call(response, target, criterion, metric_name)
                            for response, target in zip(inner_df[f'{column}_response'], inner_df[f'{column}_query'])
                        ]
                    inner_scores[f'{column}_{metric_name}'] = values

            row_scores = inner_scores.min(axis=1)
            true_positive_rows = float(row_scores.sum())
            true_positive_items = float(inner_scores.sum().sum())
            prediction_rows = len(response_df)
            gold_rows = len(answer_df)
            prediction_items = prediction_rows * len(required_columns)
            gold_items = gold_rows * len(required_columns)

            row_precision = true_positive_rows / prediction_rows if prediction_rows else 0.0
            row_recall = true_positive_rows / gold_rows if gold_rows else 0.0
            item_precision = true_positive_items / prediction_items if prediction_items else 0.0
            item_recall = true_positive_items / gold_items if gold_items else 0.0
            row_f1 = _f1(row_precision, row_recall)
            item_f1 = _f1(item_precision, item_recall)
            success_rate = float(
                row_precision == row_recall == row_f1 == 1.0 and item_precision == item_recall == item_f1 == 1.0
            )
            values = {
                'success_rate': success_rate,
                'row_precision': row_precision,
                'row_recall': row_recall,
                'row_f1': row_f1,
                'item_precision': item_precision,
                'item_recall': item_recall,
                'item_f1': item_f1,
            }
            diagnostics['stage'] = 'complete'
            return EvaluationResult(values=values, diagnostics=diagnostics)
        except Exception as error:
            diagnostics['error'] = f'{type(error).__name__}: {error}'
            return EvaluationResult(diagnostics=diagnostics)

    def _map_values(self, response: List[str], reference: List[str]) -> Tuple[Dict[str, str], str]:
        prompt = PRIMARY_KEY_PREPROCESS_PROMPT.format(response=response, reference=reference)
        judge_response = self.judge(prompt)
        parsed = parse_markdown_json(judge_response)
        if parsed is None:
            return {}, judge_response
        return {str(key): str(value) for key, value in parsed.items()}, judge_response

    def _judge_column(
        self,
        response: List[str],
        target: List[str],
        criterion: Optional[str],
    ) -> Tuple[List[float], str]:
        response_dict = {
            f'idx_{index}': {
                'response': response_value,
                'target': target_value
            }
            for index, (response_value, target_value) in enumerate(zip(response, target))
        }
        prompt = EVAL_COLUMN_PROMPT.format(criterion=criterion, response=response_dict)
        try:
            judge_response = self.judge(prompt)
        except Exception as error:
            return [0.0] * len(response), f'{type(error).__name__}: {error}'
        parsed = parse_markdown_json(judge_response)
        if parsed is None:
            return [0.0] * len(response), judge_response
        scores = []
        for index in range(len(response)):
            try:
                scores.append(float(parsed.get(f'idx_{index}', 0)))
            except (TypeError, ValueError):
                scores.append(0.0)
        return scores, judge_response

    @staticmethod
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
            all_values = [float(score.score.value[metric_name]) for score in scoped_scores]
            results.append(
                AggScore(
                    metric_name=f'{scope}/{metric_name}',
                    score=sum(all_values) / len(all_values),
                    aggregation_name=f'avg@{repeats}',
                    num=len(all_values),
                    ids=sample_ids,
                )
            )
            group_maxima = [max(float(score.score.value[metric_name]) for score in group) for group in grouped.values()]
            aggregate_name = 'pass' if metric_name == 'success_rate' else 'max'
            results.append(
                AggScore(
                    metric_name=f'{scope}/{metric_name}',
                    score=sum(group_maxima) / len(group_maxima),
                    aggregation_name=f'{aggregate_name}@{repeats}',
                    num=len(group_maxima),
                    ids=list(grouped.keys()),
                )
            )
    return results
