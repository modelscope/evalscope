# Copyright (c) Alibaba, Inc. and its affiliates.
"""minimax_verifier: multi-validator deployment-correctness benchmark.

Adapted from https://github.com/MiniMax-AI/MiniMax-Provider-Verifier .

Each prompt row carries an optional ``check_type`` list that routes it to
specific validators; rows without ``check_type`` default to tool_calls
validation. The ``error_only_reasoning`` detector is always-on across all
rows.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VendorVerifierAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import dict_to_chat_message
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.api.tool import ToolInfo
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from ._validators import (
    check_param_order_preserved,
    extract_expected_param_order,
    has_no_cyrillic_chars,
    has_no_repeated_ngram,
    validate_tool_call_with_array_command,
)

logger = get_logger()

# check_type tags from the upstream MiniMax dataset
_CT_TOOL_CALLS = 'tool_calls'
_CT_LANGUAGE = 'contains_russian_characters_unicode'
_CT_REPEAT = 'repeat_n_gram'
_CT_SCENARIO = 'scenario_check'

MINIMAX_VERIFIER_DESCRIPTION = """
## Overview

MiniMax-Vendor-Verifier is a multi-validator deployment-correctness check for MiniMax M2 / M2.5 / M2.7 vendors. Each prompt row carries an optional ``check_type`` tag that routes it through specific validators, plus an always-on ``error_only_reasoning`` detector for the most common deployment regression. Adapted from [MiniMax-Provider-Verifier](https://github.com/MiniMax-AI/MiniMax-Provider-Verifier).

## Task Description

- **Task Type**: Vendor-deployment correctness check (multi-dimensional)
- **Input**: Multi-turn chat messages with optional tool definitions, plus per-row routing tags (``check_type``, ``expected_tool_call``)
- **Output**: Vendor's chat-completion response, scored against the validator(s) selected for that row
- **Dispatch**: Rows without ``check_type`` default to the ``tool_calls`` validator; rows with ``check_type`` run only the listed validators

## Key Features

- Five upstream validators ported as pure functions:
    - ``tool_calls`` — JSON-schema validation of arguments + array-command soundness check, plus a confusion matrix over ``expected_tool_call``
    - ``error_only_reasoning`` (always-on) — flags responses with reasoning but no content and no tool calls (a deployment regression)
    - ``contains_russian_characters_unicode`` — language-following check; fails when Cyrillic codepoints leak into the response
    - ``repeat_n_gram`` — degenerate-repetition detector (any 3-gram appearing 4 or more times)
    - ``scenario_check`` — verifies the model preserves the declared JSON property order, catching providers that re-sort ``parameters.properties``
- Per-validator denominator in the report: ``num=0`` indicates no row in the subset triggered that validator (not a failure)
- Hosted dataset preserves the upstream sample.jsonl plus per-loop baseline traces for M2.5 / M2.7

## Evaluation Notes

- Default configuration uses **0-shot** evaluation; the ``default`` subset has 102 rows
- Metrics: **tool_calls_match_rate**, **schema_accuracy**, **error_only_reasoning_rate**, **language_following_success_rate**, **repeat_ngram_pass_rate**, **scenario_check_pass_rate**
- Per upstream guidance, a correctly-deployed vendor should hit ``tool_calls_match_rate ≈ 0.98``, ``schema_accuracy ≥ 0.98``, ``error_only_reasoning_rate = 0``, and ``scenario_check_pass_rate = 1.0``
- When using ``--limit``, the rarer ``check_type`` rows (scenario / repeat / language) may not all be sampled; check the per-validator ``num`` column
"""


def _decode_maybe_json(value: Any) -> Any:
    """Decode a JSON-encoded string, passing through non-strings unchanged.

    Our hosted ModelScope dataset uses empty string ``''`` to mark
    "field absent" for optional complex fields.
    """
    if value is None or value == '':
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


@register_benchmark(
    BenchmarkMeta(
        name='minimax_verifier',
        pretty_name='MiniMax-Vendor-Verifier',
        description=MINIMAX_VERIFIER_DESCRIPTION,
        tags=[Tags.FUNCTION_CALLING, Tags.AGENT],
        dataset_id='evalscope/MiniMaxVendorVerifier',
        metric_list=[
            'tool_calls_match_rate',
            'schema_accuracy',
            'error_only_reasoning_rate',
            'language_following_success_rate',
            'repeat_ngram_pass_rate',
            'scenario_check_pass_rate',
        ],
        aggregation='mean',
        subset_list=['default'],
        eval_split='test',
    )
)
class MiniMaxVerifierAdapter(VendorVerifierAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_aggregation_name = False

    # --------------------------------------------------------------------
    # Sample loading
    # --------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        messages = _decode_maybe_json(record.get('messages')) or []
        tools = _decode_maybe_json(record.get('tools')) or []
        check_type = _decode_maybe_json(record.get('check_type')) or []
        expected_tool_call = record.get('expected_tool_call')

        sample_tools: List[ToolInfo] = []
        for tool in tools or []:
            fn = tool.get('function')
            if not fn:
                continue
            # Some upstream MiniMax rows omit `description`; ToolInfo requires
            # it. Default to empty string in a shallow copy so we don't mutate
            # the original dict (preserves data quality signal for downstream
            # consumers reading `tools_raw` from metadata).
            sample_tools.append(ToolInfo.model_validate({**fn, 'description': fn.get('description', '')}))

        return Sample(
            input=[dict_to_chat_message(msg) for msg in messages],
            target='',
            tools=sample_tools,
            metadata={
                'check_type': check_type,
                'expected_tool_call': expected_tool_call,
                'tools_raw': tools,
            },
        )

    # --------------------------------------------------------------------
    # Scoring — per-row dispatch by check_type
    # --------------------------------------------------------------------

    def match_score(self, original_prediction, filtered_prediction, reference, task_state: TaskState) -> Score:
        score = Score(
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
        )
        meta = task_state.metadata
        model_output = task_state.output
        check_types: List[str] = list(meta.get('check_type') or [])
        tools_raw: List[Dict[str, Any]] = meta.get('tools_raw') or []
        expected_tool_call = meta.get('expected_tool_call')

        # Default to tool_calls when no check_type was given
        run_tool_calls = (not check_types) or (_CT_TOOL_CALLS in check_types)
        run_language = _CT_LANGUAGE in check_types
        run_repeat = _CT_REPEAT in check_types
        run_scenario = _CT_SCENARIO in check_types

        value: Dict[str, Any] = {}
        meta_out: Dict[str, Any] = {}

        # Inference-level error → only error_only_reasoning can be reasoned about
        # (we keep it as False since there's no response to inspect).
        if model_output.error:
            value['error_only_reasoning'] = 0
            value['inference_error'] = 1
            meta_out['error_reason'] = f'Model inference error: {model_output.error}'
            score.value = value
            score.metadata = meta_out
            return score

        # Always-on: error_only_reasoning
        is_reason_only = self.detect_error_only_reasoning(model_output)
        value['error_only_reasoning'] = int(is_reason_only)
        value['inference_error'] = 0

        finish_reason = model_output.stop_reason
        message = model_output.message
        tool_calls = message.tool_calls or []
        content_text = message.text or ''

        # tool_calls validator
        if run_tool_calls:
            is_call_tool = finish_reason == 'tool_calls'
            value['tool_calls_run'] = 1
            value['tool_calls_finish_tool_calls'] = int(is_call_tool)
            value['tool_calls_count'] = len(tool_calls)
            if is_call_tool and tool_calls:
                all_valid = all(validate_tool_call_with_array_command(tc, tools_raw) for tc in tool_calls)
                value['tool_calls_schema_valid'] = int(all_valid)
            else:
                value['tool_calls_schema_valid'] = 0

            # expected_tool_call confusion matrix (only when label is set)
            if expected_tool_call is True:
                value['expected_tool_call_labeled'] = 1
                value['tool_calls_match'] = int(is_call_tool)
            elif expected_tool_call is False:
                value['expected_tool_call_labeled'] = 1
                value['tool_calls_match'] = int(not is_call_tool)
            else:
                value['expected_tool_call_labeled'] = 0
                value['tool_calls_match'] = 0

        # language_following (Cyrillic absence)
        if run_language:
            value['language_following_checked'] = 1
            value['language_following_valid'] = int(has_no_cyrillic_chars(content_text))

        # repeat_n_gram
        if run_repeat:
            value['repeat_ngram_checked'] = 1
            value['repeat_ngram_valid'] = int(has_no_repeated_ngram(content_text))

        # scenario_check — order is always derived from the tools schema
        # (matches upstream ScenarioCheckValidator)
        if run_scenario:
            expected_order = extract_expected_param_order(tools_raw)
            if expected_order:
                result = check_param_order_preserved(content_text, expected_order)
                value['scenario_check_checked'] = int(result['checked'])
                value['scenario_check_valid'] = int(bool(result['valid'])) if result['checked'] else 0
                meta_out['scenario_check_detail'] = result
            else:
                value['scenario_check_checked'] = 0
                value['scenario_check_valid'] = 0

        score.value = value
        score.metadata = meta_out
        return score

    # --------------------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------------------

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        # Per-validator denominators; ``num`` on each AggScore reflects the
        # actual row count that validator was routed to (0 if it didn't run).
        # tool_calls
        tc_labeled = tc_match = 0
        tc_attempted = tc_valid = 0

        # error_only_reasoning (always-on)
        error_only = 0
        error_only_checked = 0

        # language_following
        lang_checked = lang_valid = 0

        # repeat_ngram
        rep_checked = rep_valid = 0

        # scenario_check
        sc_checked = sc_valid = 0

        for ss in sample_scores:
            v: Dict[str, Any] = ss.score.value or {}

            # error_only_reasoning is recorded for every row (set to 0 on
            # inference error so the rate isn't inflated by transport errors).
            if not v.get('inference_error'):
                error_only_checked += 1
                error_only += int(v.get('error_only_reasoning', 0))

            if v.get('tool_calls_run'):
                if v.get('tool_calls_finish_tool_calls'):
                    tc_attempted += 1
                    tc_valid += int(v.get('tool_calls_schema_valid', 0))
                if v.get('expected_tool_call_labeled'):
                    tc_labeled += 1
                    tc_match += int(v.get('tool_calls_match', 0))

            if v.get('language_following_checked'):
                lang_checked += 1
                lang_valid += int(v.get('language_following_valid', 0))

            if v.get('repeat_ngram_checked'):
                rep_checked += 1
                rep_valid += int(v.get('repeat_ngram_valid', 0))

            if v.get('scenario_check_checked'):
                sc_checked += 1
                sc_valid += int(v.get('scenario_check_valid', 0))

        def rate(num: int, den: int) -> float:
            return num / den if den else 0.0

        metrics = [
            ('tool_calls_match_rate', rate(tc_match, tc_labeled), tc_labeled),
            ('schema_accuracy', rate(tc_valid, tc_attempted), tc_attempted),
            ('error_only_reasoning_rate', rate(error_only, error_only_checked), error_only_checked),
            ('language_following_success_rate', rate(lang_valid, lang_checked), lang_checked),
            ('repeat_ngram_pass_rate', rate(rep_valid, rep_checked), rep_checked),
            ('scenario_check_pass_rate', rate(sc_valid, sc_checked), sc_checked),
        ]
        return [AggScore(metric_name=name, score=score, num=num, metadata={}) for name, score, num in metrics]
