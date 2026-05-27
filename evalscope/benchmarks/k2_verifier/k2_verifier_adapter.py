# Copyright (c) Alibaba, Inc. and its affiliates.
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

K2_VERIFIER_DESCRIPTION = """
## Overview

K2-Vendor-Verifier checks whether a third-party deployment of Kimi-K2 faithfully reproduces the official Moonshot AI API's tool-calling behavior. It replays the official evaluation prompt set against a vendor endpoint and compares finish_reason and tool-call payloads against the official baseline. Adapted from [MoonshotAI/K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier).

## Task Description

- **Task Type**: Vendor-deployment correctness check (tool calling)
- **Input**: Multi-turn chat messages with available tool definitions, identical to the upstream K2VV prompt set
- **Output**: Vendor's chat-completion response (finish_reason and tool_calls)
- **Comparison**: Vendor's behavior is compared against the official Moonshot AI baseline shipped in the dataset

## Key Features

- Uses the official 2,000-row K2-Thinking sample set (50% of the upstream test set)
- Reports the K2VV primary metric `trigger_similarity` — F1 of the tool-call decision against the official baseline
- Schema-validates triggered tool-call arguments against the declared JSON schema
- Surfaces raw counts for sanity checks (`count_finish_reason_tool_calls`, `count_successful_tool_call`)
- Hosted dataset preserves official `finish_reason` and `tool_calls` so future metrics can compare payload-level fidelity

## Evaluation Notes

- Default configuration uses **0-shot** evaluation; multi-turn context is part of each sample
- Metrics: **trigger_similarity**, **schema_accuracy**, **count_finish_reason_tool_calls**, **count_successful_tool_call**
- A `trigger_similarity` ≥ 0.73 against the official baseline is the rough acceptance threshold per the upstream K2VV README
- Only the `k2_thinking` subset is published (K2-0905 to follow when upstream releases it)
- A few historical assistant messages in the upstream baseline have malformed JSON in `tool_calls.arguments`; the adapter sanitizes them on load
"""


@register_benchmark(
    BenchmarkMeta(
        name='k2_verifier',
        pretty_name='K2-Vendor-Verifier',
        description=K2_VERIFIER_DESCRIPTION,
        tags=[Tags.FUNCTION_CALLING, Tags.AGENT],
        dataset_id='evalscope/K2VendorVerifier',
        metric_list=[
            'trigger_similarity',
            'schema_accuracy',
            'count_finish_reason_tool_calls',
            'count_successful_tool_call',
        ],
        aggregation='f1',
        subset_list=['k2_thinking'],
        eval_split='test',
    )
)
class K2VerifierAdapter(VendorVerifierAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_aggregation_name = False

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # Fields are stored as JSON-encoded strings in the hosted dataset.
        messages_raw = record.get('messages') or []
        tools_raw = record.get('tools') or []
        messages = json.loads(messages_raw) if isinstance(messages_raw, str) else messages_raw
        tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
        # The upstream K2 baseline ships a small number of assistant messages
        # whose historical tool_call arguments string is not valid JSON
        # (process.py in K2-Vendor-Verifier itself logs but does not repair
        # these). evalscope's ChatMessageAssistant strict-validates the
        # arguments field, so we sanitize on load: replace the offending
        # arguments with '{}' so the sample is still usable as context.
        _sanitize_assistant_tool_call_args(messages)
        should_call_tool = bool(record.get('should_call_tool', False))
        official_finish_reason = record.get('official_finish_reason')

        return Sample(
            input=[dict_to_chat_message(msg) for msg in messages],
            target='',
            tools=[ToolInfo.model_validate(tool['function']) for tool in tools],
            metadata={
                'should_call_tool': should_call_tool,
                'official_finish_reason': official_finish_reason,
                # Raw tool dicts kept for JSON-schema validation in match_score;
                # Sample.tools above holds the structured ToolInfo objects used
                # for model inference. Both representations are required.
                'tools': tools,
            },
        )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state: TaskState) -> Score:
        score = Score(
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
        )

        model_output = task_state.output
        should_call_tool = bool(task_state.metadata.get('should_call_tool', False))

        if model_output.error:
            score.value = {
                'finish_reason_tool_call': 0,
                'successful_tool_call': 0,
                'should_call_tool': int(should_call_tool),
            }
            score.metadata = {'error_reason': f'Model inference error: {model_output.error}'}
            return score

        finish_reason = model_output.stop_reason
        tool_calls = model_output.message.tool_calls or []
        tools = task_state.metadata['tools']

        is_call_tool = finish_reason == 'tool_calls'
        if is_call_tool:
            # Surface tool call turns in the rendered prediction for review UI
            score.prediction = task_state.messages_markdown
            score.extracted_prediction = task_state.messages_markdown

        is_valid_tool_call, error_reason = self.validate_tool_call(tool_calls, tools)
        is_call_successful = is_call_tool and is_valid_tool_call

        score.value = {
            'finish_reason_tool_call': int(is_call_tool),
            'successful_tool_call': int(is_call_successful),
            'should_call_tool': int(should_call_tool),
        }
        if error_reason:
            score.metadata = {'error_reason': error_reason}
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Compute K2-Vendor-Verifier metrics.

        - trigger_similarity: F1 of (vendor predicted tool_calls) vs (official tool_calls label).
        - schema_accuracy: among vendor tool-call attempts, fraction passing JSON schema.
        - count_finish_reason_tool_calls / count_successful_tool_call: raw counts.
        """
        finish_reason_tool_call_count = 0
        successful_tool_call_count = 0
        tp = fp = fn = 0
        total = len(sample_scores)

        for ss in sample_scores:
            v = ss.score.value or {}
            pred = int(v.get('finish_reason_tool_call', 0))
            ok = int(v.get('successful_tool_call', 0))
            label = int(v.get('should_call_tool', 0))

            finish_reason_tool_call_count += pred
            successful_tool_call_count += ok

            if pred and label:
                tp += 1
            elif pred and not label:
                fp += 1
            elif not pred and label:
                fn += 1

        # pred ∈ {0,1}, so finish_reason_tool_call_count == attempted tool calls,
        # and successful_tool_call_count == those that also passed schema.
        schema_accuracy = (
            successful_tool_call_count / finish_reason_tool_call_count
        ) if finish_reason_tool_call_count else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        trigger_similarity = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics = {
            'trigger_similarity': trigger_similarity,
            'schema_accuracy': schema_accuracy,
            'count_finish_reason_tool_calls': finish_reason_tool_call_count,
            'count_successful_tool_call': successful_tool_call_count,
        }
        return [AggScore(metric_name=name, score=val, num=total, metadata={}) for name, val in metrics.items()]


def _sanitize_assistant_tool_call_args(messages: List[Dict[str, Any]]) -> None:
    """Replace malformed ``tool_calls[*].function.arguments`` strings with ``'{}'``.

    A small number of upstream K2 baseline rows ship historical assistant
    messages whose tool-call arguments string is not valid JSON. ChatMessage
    Assistant strict-validates this field, so we sanitize in place to allow
    the sample to load. The malformed args are only used as conversational
    context, not for scoring.
    """
    for msg in messages:
        if msg.get('role') != 'assistant':
            continue
        for tc in msg.get('tool_calls') or []:
            fn = tc.get('function') or {}
            args = fn.get('arguments')
            if not isinstance(args, str):
                continue
            try:
                json.loads(args)
            except (json.JSONDecodeError, ValueError):
                fn['arguments'] = '{}'
