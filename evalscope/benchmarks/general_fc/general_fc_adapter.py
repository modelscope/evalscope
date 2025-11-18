# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import dict_to_chat_message
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.model.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.api.tool import ToolCall, ToolInfo
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='general_fc',
        pretty_name='General-FunctionCalling',
        description='A general function calling dataset for custom evaluation. '
        'For detailed instructions on how to use this benchmark, please refer to the [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#fc).',  # noqa: E501
        tags=[Tags.FUNCTION_CALLING, Tags.CUSTOM, Tags.AGENT],
        dataset_id='evalscope/GeneralFunctionCall-Test',
        metric_list=[
            'count_finish_reason_tool_call',
            'count_successful_tool_call',
            'schema_accuracy',
            'tool_call_f1',
        ],
        aggregation='f1',
        eval_split='test',
    )
)
class GeneralFCAdapter(AgentAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_aggregation_name = False

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        messages = record.get('messages', [])
        tools = record.get('tools', [])
        should_call_tool = record.get('should_call_tool', False)

        # In case the fields are stored as JSON strings
        if isinstance(messages, str):
            messages = json.loads(messages)
        if isinstance(tools, str):
            tools = json.loads(tools)

        # Convert to Sample
        return Sample(
            input=[dict_to_chat_message(msg) for msg in messages],
            target='',
            tools=[ToolInfo.model_validate(tool['function']) for tool in tools],
            metadata={
                'should_call_tool': should_call_tool,
                'tools': tools,
            }
        )

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        # Execute model inference with the processed input and any tools
        try:
            model_output = model.generate(input=sample.input, tools=sample.tools)
            return model_output
        except Exception as e:
            logger.error(f'Error during model inference: {e}')
            return ModelOutput.from_content(
                content='',
                stop_reason='stop',
                error=str(e),
            )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state: TaskState) -> Score:
        score = Score(
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
        )

        model_output = task_state.output
        should_call_tool = task_state.metadata.get('should_call_tool', False)
        if model_output.error:
            score.value = {
                'finish_reason_tool_call': 0,
                'successful_tool_call': 0,
                'should_call_tool': int(should_call_tool),
            }
            score.metadata = {
                'error_reason': f'Model inference error: {model_output.error}',
            }

        else:
            finish_reason = model_output.stop_reason
            tool_calls = model_output.message.tool_calls or []
            tools = task_state.metadata['tools']

            is_call_tool = finish_reason == 'tool_calls'
            is_valid_tool_call, error_reason = self.validate_tool_call(tool_calls, tools)
            is_call_successful = is_call_tool and is_valid_tool_call
            score.value = {
                'finish_reason_tool_call': int(is_call_tool),
                'successful_tool_call': int(is_call_successful),
                'should_call_tool': int(should_call_tool),
            }
            score.metadata = {
                'error_reason': error_reason,
            }

        return score

    @staticmethod
    def validate_tool_call(tool_calls: List[ToolCall], tools: List[Dict[str, Any]]) -> Tuple[bool, str]:
        from jsonschema import ValidationError, validate

        try:
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                # Find corresponding tool schema
                schema = next(
                    (t['function']['parameters'] for t in tools if t['function']['name'] == tool_name),
                    None,
                )
                if not schema:
                    return False, f"No schema found for tool '{tool_name}'"

                # Parse arguments (may be string or dict)
                args = tool_call.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError as e:
                        return False, f"JSON parse failed for tool '{tool_name}' arguments: {e}"

                # Validate using jsonschema
                validate(instance=args, schema=schema)

        except ValidationError as e:
            return False, f"Schema validation failed for tool '{tool_name}': {e.message}"
        except KeyError as e:
            return False, f'Tool call format error, missing field: {e}'
        except Exception as e:
            return False, f'Unexpected error during validation: {e}'
        return True, ''

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Aggregate General-FunctionCalling metrics:
        - count_finish_reason_tool_call: number of samples with finish_reason == 'tool_calls'.
        - count_successful_tool_call: number of schema-valid tool calls among attempts.
        - schema_accuracy: valid tool calls / attempted tool calls (0.0 if no attempts).
        - tool_call_f1: F1 over whether a tool should be called vs. model predicted tool call.
        """
        finish_reason_tool_call_count = 0
        successful_tool_call_count = 0
        attempted_tool_calls = 0
        valid_tool_calls = 0

        tp = fp = fn = tn = 0
        total_count = len(sample_scores)

        for ss in sample_scores:
            # Values set in match_score
            v = ss.score.value or {}
            pred_is_tool = int(v.get('finish_reason_tool_call', 0))  # model predicted to call tool
            pred_is_success = int(v.get('successful_tool_call', 0))  # model called tool and schema validated
            label_should_tool = int(v.get('should_call_tool', 0))  # ground-truth label

            # Counts
            finish_reason_tool_call_count += pred_is_tool
            successful_tool_call_count += pred_is_success

            if pred_is_tool == 1:
                attempted_tool_calls += 1
                if pred_is_success == 1:
                    valid_tool_calls += 1

            # Confusion matrix for F1
            if pred_is_tool == 1 and label_should_tool == 1:
                tp += 1
            elif pred_is_tool == 1 and label_should_tool == 0:
                fp += 1
            elif pred_is_tool == 0 and label_should_tool == 1:
                fn += 1
            else:
                tn += 1

        # Schema accuracy among attempts
        schema_accuracy = (valid_tool_calls / attempted_tool_calls) if attempted_tool_calls > 0 else 0.0

        # Precision/Recall/F1 for tool-calling decision
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        tool_call_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics = {
            'count_finish_reason_tool_call': finish_reason_tool_call_count,
            'count_successful_tool_call': successful_tool_call_count,
            'schema_accuracy': schema_accuracy,
            'tool_call_f1': tool_call_f1,
        }

        agg_scores: List[AggScore] = []
        for name, value in metrics.items():
            agg_scores.append(AggScore(metric_name=name, score=value, num=total_count, metadata={}))

        return agg_scores
