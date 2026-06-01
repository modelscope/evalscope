# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage, ChatMessageSystem, dict_to_chat_message
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from .utils import (
    build_tool_infos,
    decode_maybe_json,
    extract_tool_calls_from_output,
    parse_call_list,
    score_agent_call,
    score_normal_call,
    score_special_call,
    split_acebench_messages,
)

ACEBENCH_SUBSETS = ['normal', 'special', 'agent']


@register_benchmark(
    BenchmarkMeta(
        name='acebench',
        pretty_name='ACEBench',
        tags=[Tags.FUNCTION_CALLING, Tags.AGENT, Tags.MULTI_TURN],
        description="""
## Overview

ACEBench is a tool-use benchmark for evaluating whether large language models can select APIs, fill
arguments, handle abnormal requests, and complete realistic agent tasks.

## Task Description

- **Task Type**: Function calling and agentic tool use
- **Input**: Conversation history, API specifications, optional time/profile context, and agent task context
- **Output**: Function calls or diagnostic text for special cases
- **Subsets**: normal, special, and agent

## Evaluation Notes

- The adapter passes ACEBench API specifications as EvalScope tools and also includes concise text
  instructions for text-only models.
- Normal samples are scored by matching function names and arguments.
- Special samples are scored against ACEBench's diagnostic text contract.
- Agent samples report `process_acc` against ACEBench milestones. If a model returns a final-state JSON object,
  `end_state_acc` is also reported and used as `acc`; otherwise `acc` follows `process_acc`.
""",
        dataset_id='evalscope/acebench',
        subset_list=ACEBENCH_SUBSETS,
        default_subset='en',
        metric_list=['acc', 'process_acc', 'end_state_acc'],
        eval_split='normal',
    )
)
class AceBenchAdapter(DefaultDataAdapter):
    """ACEBench adapter using EvalScope's default single-turn data path."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.split_as_subset = True
        self.add_aggregation_name = False

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert an ACEBench record to an EvalScope Sample."""
        functions = decode_maybe_json(record.get('function'), [])
        rubric = decode_maybe_json(record.get('rubric'), {})
        ground_truth = rubric.get('ground_truth', {})
        milestones = rubric.get('mile_stone', [])
        sub_category = record.get('sub_category') or _category_from_id(record.get('id') or '')

        messages = self._build_input_messages(record=record, functions=functions, sub_category=sub_category)
        target = json.dumps({'ground_truth': ground_truth, 'mile_stone': milestones}, ensure_ascii=False)

        return Sample(
            input=messages,
            target=target,
            tools=build_tool_infos(functions),
            metadata={
                'id': record.get('id'),
                'sub_category': sub_category,
                'question': record.get('question', ''),
                'time': record.get('time', ''),
                'profile': record.get('profile', ''),
                'functions': functions,
                'ground_truth': ground_truth,
                'mile_stone': milestones,
                'initial_config': decode_maybe_json(record.get('initial_config'), {}),
                'involved_classes': decode_maybe_json(record.get('involved_classes'), []),
            },
        )

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        metadata = task_state.metadata or {}
        sub_category = metadata.get('sub_category', '')
        predicted_calls = extract_tool_calls_from_output(task_state.output) or parse_call_list(filtered_prediction)

        if 'special' in sub_category:
            result = score_special_call(filtered_prediction, metadata.get('ground_truth'), sub_category)
        elif 'agent' in sub_category:
            result = score_agent_call(
                prediction=filtered_prediction,
                predicted_calls=predicted_calls,
                expected_answer=metadata.get('ground_truth'),
                milestones=metadata.get('mile_stone'),
            )
        else:
            result = score_normal_call(
                predicted_calls=predicted_calls,
                expected_answer=metadata.get('ground_truth'),
                test_category=sub_category,
            )

        score.value = {
            key: value
            for key, value in result.items()
            if key in {'acc', 'process_acc', 'end_state_acc'} and value is not None
        }
        score.metadata = {
            'valid': result.get('valid', False),
            'error': result.get('error', ''),
            'error_type': result.get('error_type', ''),
            'predicted_calls': predicted_calls,
        }
        extra_metadata = result.get('metadata')
        if isinstance(extra_metadata, dict):
            score.metadata.update(extra_metadata)
        score.main_score_name = 'acc'
        return score

    @staticmethod
    def _build_input_messages(
        record: Dict[str, Any],
        functions: List[Dict[str, Any]],
        sub_category: str,
    ) -> List[ChatMessage]:
        system_prompt = _build_system_prompt(record=record, functions=functions, sub_category=sub_category)
        messages = [ChatMessageSystem(content=system_prompt)]
        messages.extend(
            dict_to_chat_message(message) for message in split_acebench_messages(record.get('question', ''))
        )
        return messages


def _build_system_prompt(record: Dict[str, Any], functions: List[Dict[str, Any]], sub_category: str) -> str:
    function_specs = json.dumps(functions, ensure_ascii=False)
    context_parts = [
        'You are evaluating ACEBench tool-use tasks.',
        'Use the available tool schemas when native function calling is supported.',
        (
            'For text-only output, return API calls as [ApiName(key1=\'value1\', key2=2)]. '
            'Return only the call list and no extra explanation.'
        ),
    ]

    if 'special' in sub_category:
        context_parts.append(
            'For incomplete, incorrect, or unsupported requests, do not call tools. Return the required diagnostic '
            'string exactly enough to identify the missing parameters, incorrect values, or function limitation.'
        )
    elif 'agent' in sub_category:
        initial_config = record.get('initial_config') or '{}'
        involved_classes = record.get('involved_classes') or '[]'
        context_parts.append(
            'For agent tasks, plan the needed tool-call sequence. If returning text, output the calls in execution order.'
        )
        context_parts.append(f'Initial configuration: {initial_config}')
        context_parts.append(f'Involved classes: {involved_classes}')
    elif 'preference' in sub_category and record.get('profile'):
        context_parts.append(f'Character profile: {record["profile"]}')

    if record.get('time'):
        context_parts.append(str(record['time']))

    context_parts.append(f'API specifications: {function_specs}')
    return '\n\n'.join(context_parts)


def _category_from_id(record_id: str) -> str:
    if record_id.startswith('normal_multi_turn'):
        return '_'.join(record_id.split('_')[:-2])
    return record_id.rsplit('_', 1)[0] if '_' in record_id else record_id
