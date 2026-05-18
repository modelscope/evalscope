import json
import os
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.dataset.dataset import DatasetDict
from evalscope.api.dataset.loader import DictDataLoader
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages.chat_message import ChatMessageAssistant, ChatMessageUser
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.report import Report
from evalscope.utils.function_utils import thread_safe
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger
from .utils import (
    ALL_SCORING_CATEGORIES,
    compute_aggregate_subsets,
    compute_entry_result,
    load_bfcl_data,
    process_test_entries,
    run_prereq_inference,
    synthesize_agent_messages,
)

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='bfcl_v4',
        pretty_name='BFCL-v4',
        tags=[Tags.FUNCTION_CALLING, Tags.AGENT],
        description="""
## Overview

BFCL-v4 (Berkeley Function-Calling Leaderboard V4) is a comprehensive benchmark for evaluating agentic function-calling capabilities of LLMs. It tests web search, memory operations, and format sensitivity as building blocks for agentic applications.

## Task Description

- **Task Type**: Agentic Function Calling Evaluation
- **Input**: Scenarios requiring function calls for web search, memory, etc.
- **Output**: Correct function calls with proper arguments
- **Capabilities**: Web search, memory read/write, format handling

## Key Features

- Holistic agentic evaluation framework
- Tests building blocks for LLM agents (search, memory, formatting)
- Multiple scoring categories for detailed analysis
- Supports both FC models and non-FC models
- Optional web search with SerpAPI integration

## Evaluation Notes

- **Installation Required**: `pip install bfcl-eval==2025.10.27.1`
- Primary metric: **Accuracy** per category
- Configure `is_fc_model` based on model capabilities
- Optional: Set `SERPAPI_API_KEY` for web search tasks
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v4.html)
""",
        dataset_id='https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard',
        subset_list=ALL_SCORING_CATEGORIES,
        metric_list=['acc'],
        eval_split='train',
        extra_params={
            'underscore_to_dot': {
                'type': 'bool',
                'description': 'Convert underscores to dots in function names for evaluation.',
                'value': True
            },
            'is_fc_model': {
                'type': 'bool',
                'description': 'Indicates the evaluated model natively supports function calling.',
                'value': True
            },
            'SERPAPI_API_KEY': {
                'type': 'str | null',
                'description': 'SerpAPI key enabling web-search capability in BFCL V4. Null disables web search.',
                'value': None
            }
        }
    )
)
class BFCLV4Adapter(AgentAdapter):
    """
    BFCL adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import('bfcl_eval', extra='bfcl', raise_error=True, feature_name=self.pretty_name)

        self.add_overall_metric = False
        self.add_aggregation_name = False

        self.underscore_to_dot = self.extra_params.get('underscore_to_dot', True)
        self.is_fc_model = self.extra_params.get('is_fc_model', True)
        # Set SERPAPI_API_KEY in environment variables if provided
        serpapi_api_key = self.extra_params.get('SERPAPI_API_KEY', None)
        if serpapi_api_key:
            os.environ['SERPAPI_API_KEY'] = serpapi_api_key
        self.model_result_dir = Path(self._task_config.work_dir) if self._task_config else Path('./bfcl_model_results')
        self.handler = None
        self.prereq_entries = []
        self.prereq_finished = False

    def load(self):
        """Load and process the BFCL dataset."""
        from bfcl_eval.utils import parse_test_category_argument
        datasets = {}
        all_test_categories = parse_test_category_argument(self.subset_list)

        test_entries_by_cat, ground_truth_by_cat = load_bfcl_data(all_test_categories)

        for category in all_test_categories:
            test_entries = test_entries_by_cat.get(category, [])
            ground_truth_entries = ground_truth_by_cat.get(category, [])

            if not test_entries:
                continue

            datasets[category] = self._create_dataset_for_category(category, test_entries, ground_truth_entries)

        test_dataset = DatasetDict(datasets)
        return test_dataset, None

    def _create_dataset_for_category(
        self, category: str, test_entries: List[Dict], ground_truth_entries: List[Dict]
    ) -> DatasetDict:
        """Create a dataset for a single category by merging test and ground truth data."""
        processed_entries, prereq_entries = process_test_entries(
            category=category,
            test_entries=test_entries,
            ground_truth_entries=ground_truth_entries,
            model_result_dir=self.model_result_dir,
        )
        # collect prereq entries for later prereq inference
        self.prereq_entries.extend(prereq_entries)

        return DictDataLoader(
            dict_list=processed_entries,
            limit=self.limit,
            repeats=self.repeats,
            sample_fields=self.record_to_sample,
            shuffle=self.shuffle,
        ).load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        return Sample(
            input=[ChatMessageUser(content=json.dumps(record['question']))],
            target=json.dumps(record['ground_truth']),  # Will use the record for evaluation
            metadata=record  # Store the full record for evaluation
        )

    @thread_safe
    def _init_handler(self):
        if self.handler is not None:
            return  # Handler already initialized

        from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

        # Set env variables for OpenAI API
        os.environ['OPENAI_API_KEY'] = self._task_config.api_key
        os.environ['OPENAI_BASE_URL'] = self._task_config.api_url

        self.handler = OpenAICompletionsHandler(
            model_name=self._task_config.model,
            temperature=self._task_config.generation_config.temperature,
            registry_name=self._task_config.model_id,
            is_fc_model=self.is_fc_model,
        )

        self._prereq_inference()

    def _prereq_inference(self):
        if self.prereq_finished:
            return
        run_prereq_inference(
            handler=self.handler,
            prereq_entries=self.prereq_entries,
            model_result_dir=self.model_result_dir,
            batch_size=self._task_config.eval_batch_size,
            logger=logger,
        )
        self.prereq_finished = True

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        try:
            self._init_handler()

            result, metadata = self.handler.inference(
                deepcopy(sample.metadata), include_input_log=False, exclude_state_log=True
            )

            output = ModelOutput.from_content(
                model=model.name,
                content=json.dumps(result),
            )
            agent_messages = synthesize_agent_messages(
                prompt_entry=sample.metadata,
                model_result=result,
                is_fc_model=self.is_fc_model,
                inference_log=metadata.get('inference_log') if isinstance(metadata, dict) else None,
            )
            self._attach_perf_metrics(output, agent_messages, metadata)
            return InferenceResult(output=output, messages=agent_messages or None)
        except Exception as e:
            # This is usually the case when the model getting stuck on one particular test case.
            # For example, timeout error or FC model returning invalid JSON response.
            # Since temperature is already set to 0.001, retrying the same test case will not help.
            # So we continue the generation process and record the error message as the model response
            logger.error(f'Error during inference for sample ID {sample.metadata.get("id")}: {e}')
            logger.error(traceback.format_exc())

            return ModelOutput.from_content(
                model=model.name,
                content=json.dumps({
                    'error': str(e),
                    'error_message': traceback.format_exc(),
                }),
            )

    @staticmethod
    def _attach_perf_metrics(output: ModelOutput, agent_messages, metadata) -> None:
        """Translate bfcl handler `metadata` into per-step PerformanceMetrics.

        Single-turn metadata holds scalar latency / token counts; multi-turn
        holds list[list[float]] indexed by [turn][step]. Each ChatMessageAssistant
        in `agent_messages` corresponds positionally to one (turn, step), so we
        attach one PerformanceMetrics per assistant message. The full call's
        totals are also stored on `output.message.perf_metrics` for aggregate
        reporting. TTFT is unavailable (handler is non-streaming).
        """
        if not isinstance(metadata, dict):
            return
        latency = metadata.get('latency')
        if latency is None:
            return
        in_tok = metadata.get('input_token_count')
        out_tok = metadata.get('output_token_count')

        def _flatten(x):
            if isinstance(x, list):
                for y in x:
                    yield from _flatten(y)
            else:
                yield x

        if isinstance(latency, list):
            lat_list = list(_flatten(latency))
            in_list = list(_flatten(in_tok)) if isinstance(in_tok, list) else [in_tok] * len(lat_list)
            out_list = list(_flatten(out_tok)) if isinstance(out_tok, list) else [out_tok] * len(lat_list)
        else:
            lat_list = [latency]
            in_list = [in_tok if in_tok is not None else 0]
            out_list = [out_tok if out_tok is not None else 0]

        idx = 0
        for msg in agent_messages or []:
            if idx >= len(lat_list):
                break
            if isinstance(msg, ChatMessageAssistant):
                msg.perf_metrics = PerformanceMetrics(
                    latency=float(lat_list[idx] or 0.0),
                    ttft=None,
                    input_tokens=int(in_list[idx] or 0),
                    output_tokens=int(out_list[idx] or 0),
                )
                idx += 1

        total_lat = float(sum(l or 0.0 for l in lat_list))
        output.message.perf_metrics = PerformanceMetrics(
            latency=total_lat,
            ttft=None,
            input_tokens=int(sum(i or 0 for i in in_list)),
            output_tokens=int(sum(o or 0 for o in out_list)),
        )
        output.time = total_lat

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        self._init_handler()

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        model_result = json.loads(filtered_prediction)
        prompt = task_state.metadata

        entry_result = compute_entry_result(
            handler=self.handler,
            model_result=model_result,
            prompt_entry=prompt,
            underscore_to_dot=self.underscore_to_dot,
        )

        valid = 1 if entry_result['valid'] else 0
        score.value = {'acc': valid}
        score.metadata = {
            'valid': bool(entry_result.get('valid')),
            'error': str(entry_result.get('error')),
            'error_message': str(entry_result.get('error_message')),
            'error_type': str(entry_result.get('error_type')),
        }
        return score

    def _on_generate_report_end(self, report: Report, output_dir, **kwargs):
        """
        Finalize the report generation process. Calculate the overall score.
        """

        # noqa: E501
        compute_aggregate_subsets(report)
