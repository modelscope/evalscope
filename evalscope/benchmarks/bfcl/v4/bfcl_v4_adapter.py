import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample, build_dataset_from_records
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages.chat_message import ChatMessageAssistant, ChatMessageUser
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoreStatus, Tags
from evalscope.report import Report
from evalscope.utils.argument_utils import get_secret_value
from evalscope.utils.function_utils import thread_safe
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

from .utils import (
    ALL_SCORING_CATEGORIES,
    compute_aggregate_subsets,
    compute_entry_result,
    load_bfcl_data,
    patch_openai_completions_handler_empty_tool_calls,
    process_test_entries,
    run_isolated_inference,
    run_prereq_inference,
    synthesize_agent_messages,
)

logger = get_logger()


def _normalize_openai_base_url(api_url: Optional[str]) -> str:
    """Return the API root expected by the OpenAI SDK."""
    if not api_url:
        return ''
    return api_url.strip().rstrip('/').removesuffix('/chat/completions')


@register_benchmark(
    BenchmarkMeta(
        name='bfcl_v4',
        pretty_name='BFCL-v4',
        evaluation_version='v1.2',
        tags=[Tags.FUNCTION_CALLING, Tags.AGENT],
        description="""
## Overview

BFCL-v4 (Berkeley Function-Calling Leaderboard V4) is a comprehensive benchmark for evaluating agentic function-calling capabilities of LLMs. It tests web search, memory operations, and format sensitivity as building blocks for agentic applications.

## Task Description

- **Task Type**: Agentic Function Calling Evaluation
- **Input**: Scenarios requiring function calls for web search, memory, etc.
- **Output**: Function calls with arguments and final answers for agentic tasks
- **Capabilities**: Web search, memory read/write, format handling

## Key Features

- Holistic agentic evaluation framework
- Tests building blocks for LLM agents (search, memory, formatting)
- Multiple scoring categories for detailed analysis
- Supports both FC models and non-FC models
- Optional web search with SerpAPI integration

## Evaluation Notes

- **Installation Required**: `pip install 'evalscope[bfcl]'` with `bfcl-eval==2026.3.23`
- Primary metric: **Accuracy** per category
- Configure `is_fc_model` based on model capabilities
- Memory tasks prepare the scenarios required by the selected samples. An empty memory snapshot is logged as a warning
- Web search requires a valid `SERPAPI_API_KEY`. To evaluate without a key, omit `web_search_base` and `web_search_no_snippet` from `subset_list`
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
                'value': True,
            },
            'is_fc_model': {
                'type': 'bool',
                'description': 'Indicates the evaluated model natively supports function calling.',
                'value': True,
            },
            'SERPAPI_API_KEY': {
                'type': 'str | null',
                'description': 'SerpAPI key for web search. If unset, BFCL reads the SERPAPI_API_KEY environment variable.',
                'value': None,
            },
        },
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

        self.underscore_to_dot = self.extra_params.get('underscore_to_dot', True)
        self.is_fc_model = self.extra_params.get('is_fc_model', True)
        # Set SERPAPI_API_KEY in environment variables if provided
        serpapi_api_key = self.extra_params.get('SERPAPI_API_KEY', None)
        if serpapi_api_key:
            os.environ['SERPAPI_API_KEY'] = serpapi_api_key
        self.model_result_dir = Path(self._task_config.work_dir) if self._task_config else Path('./bfcl_model_results')
        self.handler = None
        self.prereq_entries = []
        self.prereq_failures: set[tuple[str, str]] = set()
        self.prereq_finished = False

    def load(self):
        """Load and process the BFCL dataset."""
        from bfcl_eval.utils import parse_test_category_argument

        datasets = {}
        all_test_categories = parse_test_category_argument(self.subset_list)

        self._warn_if_web_search_unconfigured(all_test_categories)

        test_entries_by_cat, ground_truth_by_cat = load_bfcl_data(all_test_categories)

        for category in all_test_categories:
            test_entries = test_entries_by_cat.get(category, [])
            ground_truth_entries = ground_truth_by_cat.get(category, [])

            if not test_entries:
                continue

            datasets[category] = self._create_dataset_for_category(category, test_entries, ground_truth_entries)

        test_dataset = DatasetDict(datasets)
        return test_dataset, None

    def _warn_if_web_search_unconfigured(self, categories: List[str]) -> None:
        """Warn when web-search subsets are selected without a SerpAPI key.

        Without ``SERPAPI_API_KEY`` every search call returns an error payload
        instead of results, so the model answers against an error and the
        subsets score 0 — after the full, expensive run. The adapter documents
        omitting these subsets, but a user who keeps the default subset list has
        no way to notice the missing key until the run ends. Warn up front so
        the gap is visible before any tokens are spent.
        """
        from bfcl_eval.utils import is_web_search

        web_search_subsets = [category for category in categories if is_web_search(category)]
        if not web_search_subsets or os.environ.get('SERPAPI_API_KEY'):
            return
        logger.warning(
            f'BFCL web-search subsets {web_search_subsets} are selected but SERPAPI_API_KEY is not set. '
            'Every search call will return an error instead of results, so these subsets will score 0 '
            'regardless of the model. Set SERPAPI_API_KEY, or omit these subsets from subset_list.'
        )

    def _create_dataset_for_category(
        self, category: str, test_entries: List[Dict], ground_truth_entries: List[Dict]
    ) -> MemoryDataset:
        """Create a dataset for a single category by merging test and ground truth data."""
        processed_entries, prereq_entries = process_test_entries(
            category=category,
            test_entries=test_entries,
            ground_truth_entries=ground_truth_entries,
            model_result_dir=self.model_result_dir,
        )
        dataset = build_dataset_from_records(
            records=processed_entries,
            sample_fields=self.record_to_sample,
            name=category,
            location=self.dataset_id,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self._task_config.seed if self._task_config else None,
        )
        if prereq_entries:
            required_ids = {entry_id for sample in dataset for entry_id in sample.metadata['depends_on']}
            self.prereq_entries.extend(entry for entry in prereq_entries if entry['id'] in required_ids)
        return dataset

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        return Sample(
            input=[ChatMessageUser(content=json.dumps(record['question']))],
            target=json.dumps(record['ground_truth']),  # Will use the record for evaluation
            metadata=record,  # Store the full record for evaluation
        )

    @thread_safe
    def _init_handler(self) -> None:
        if self.handler is not None:
            return  # Handler already initialized

        from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

        base_url = _normalize_openai_base_url(self._task_config.api_url)
        api_key = get_secret_value(self._task_config.api_key) or ''
        generation = self._task_config.generation_config

        class ConfiguredHandler(OpenAICompletionsHandler):
            def _build_client_kwargs(self) -> Dict[str, Any]:
                kwargs = super()._build_client_kwargs()
                kwargs.update(api_key=api_key, base_url=base_url or 'https://api.openai.com/v1')
                if generation.timeout is not None:
                    kwargs['timeout'] = generation.timeout
                return kwargs

            def generate_with_backoff(self, **kwargs: Any) -> Any:
                if generation.extra_body:
                    kwargs['extra_body'] = generation.extra_body
                return super().generate_with_backoff(**kwargs)

        patch_openai_completions_handler_empty_tool_calls(ConfiguredHandler)
        self.handler = ConfiguredHandler(
            model_name=self._task_config.model,
            temperature=self._task_config.generation_config.temperature,
            registry_name=self._task_config.model_id,
            is_fc_model=self.is_fc_model,
        )

    @thread_safe
    def _prereq_inference(self) -> None:
        if self.prereq_finished:
            return
        self.prereq_failures = run_prereq_inference(
            handler=self.handler,
            prereq_entries=self.prereq_entries,
            model_result_dir=self.model_result_dir,
            batch_size=self._task_config.eval_batch_size,
            logger=logger,
        )
        self.prereq_finished = True

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        self._init_handler()
        self._prereq_inference()
        category = sample.metadata['category']
        scenario = sample.metadata.get('scenario')
        if (category.removeprefix('memory_'), scenario) in self.prereq_failures:
            raise RuntimeError(f'BFCL memory prerequisites did not complete for {category}/{scenario}')

        result, metadata = run_isolated_inference(
            self.handler, sample.metadata, include_input_log=False, exclude_state_log=True
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

        total_lat = float(sum(latency or 0.0 for latency in lat_list))
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
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        model_result = json.loads(filtered_prediction)
        if isinstance(model_result, dict) and 'error' in model_result:
            score.status = ScoreStatus.INVALID_SESSION
            score.metadata = model_result
            return score

        self._init_handler()
        prompt = task_state.metadata

        entry_result = compute_entry_result(
            handler=self.handler,
            model_result=model_result,
            prompt_entry=prompt,
            underscore_to_dot=self.underscore_to_dot,
        )

        valid = 1 if entry_result['valid'] else 0
        score.value = {'acc': valid}
        error = entry_result.get('error')
        details = error if isinstance(error, dict) else entry_result
        score.metadata = {
            'valid': bool(entry_result.get('valid')),
            'error': str(error),
            'error_message': str(details.get('error_message')),
            'error_type': str(details.get('error_type')),
        }
        return score

    def _on_generate_report_end(self, report: Report, output_dir, **kwargs):
        """
        Finalize the report generation process. Calculate the overall score.
        """

        # noqa: E501
        compute_aggregate_subsets(report)
