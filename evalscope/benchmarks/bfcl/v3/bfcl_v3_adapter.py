import json
import traceback
from typing import Any, Dict, List

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.report import Category, Report, Subset, unweighted_average_from_subsets, weighted_average_from_subsets
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBJECT_MAPPING = {
    'simple': 'AST_NON_LIVE',
    'multiple': 'AST_NON_LIVE',
    'parallel': 'AST_NON_LIVE',
    'parallel_multiple': 'AST_NON_LIVE',
    'java': 'AST_NON_LIVE',
    'javascript': 'AST_NON_LIVE',
    'live_simple': 'AST_LIVE',
    'live_multiple': 'AST_LIVE',
    'live_parallel': 'AST_LIVE',
    'live_parallel_multiple': 'AST_LIVE',
    'irrelevance': 'RELEVANCE',
    'live_relevance': 'RELEVANCE',
    'live_irrelevance': 'RELEVANCE',
    'multi_turn_base': 'MULTI_TURN',
    'multi_turn_miss_func': 'MULTI_TURN',
    'multi_turn_miss_param': 'MULTI_TURN',
    'multi_turn_long_context': 'MULTI_TURN'
}

BFCL_V3_TO_V4_SUBJECT_MAPPING = {
    'simple': 'simple_python',
    'java': 'simple_java',
    'javascript': 'simple_javascript',
}


@register_benchmark(
    BenchmarkMeta(
        name='bfcl_v3',
        pretty_name='BFCL-v3',
        tags=[Tags.FUNCTION_CALLING, Tags.AGENT],
        description='Berkeley Function Calling Leaderboard (BFCL), the **first comprehensive '
        'and executable function call evaluation** '
        'dedicated to assessing Large Language Models\' (LLMs) ability to invoke '
        'functions. Unlike previous evaluations, '
        'BFCL accounts for various forms of function calls, diverse scenarios, and executability. '
        'Need to run `pip install bfcl-eval==2025.10.27.1` before evaluating. '
        '[Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html)',
        dataset_id='AI-ModelScope/bfcl_v3',
        subset_list=list(SUBJECT_MAPPING.keys()),
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
            }
        }
    )
)
class BFCLV3Adapter(AgentAdapter):
    """
    BFCL adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import('bfcl_eval', package='bfcl-eval==2025.10.27.1', raise_error=True, feature_name=self.pretty_name)

        self.category_map = SUBJECT_MAPPING
        self.reformat_subset = True
        self.add_overall_metric = False
        self.add_aggregation_name = False

        self.underscore_to_dot = self.extra_params.get('underscore_to_dot', True)
        self.is_fc_model = self.extra_params.get('is_fc_model', True)

    def preprocess_row(self, row: dict):
        """
        Inplace preprocess the row to ensure it has the correct format for BFCL evaluation.
        """
        row['should_execute_tool_calls'] = True if row['multi_turn'] else False
        row['functions'] = json.loads(row['functions'])
        row['tools'] = json.loads(row['tools'])
        row['turns'] = json.loads(row['turns'])
        row['missing_functions'] = json.loads(row['missed_functions'])
        row['ground_truth'] = json.loads(row.get('ground_truth', '{}'))
        row['initial_config'] = json.loads(row['initial_config'])
        row['is_fc_model'] = self.is_fc_model

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        self.preprocess_row(record)

        # If the model is a function calling model, we need to remove the system prompt
        if self.is_fc_model:
            turns = record['turns']
            new_turns = []
            for turn_idx, messages in enumerate(turns):
                current_messages = messages.copy()
                if len(current_messages) > 0 and current_messages[0]['role'] == 'system':
                    current_messages = current_messages[1:]
                new_turns.append(current_messages)
            record['turns'] = new_turns

        return Sample(
            input=[ChatMessageUser(content=json.dumps(record['turns']))],
            target=json.dumps(record['ground_truth']),  # Will use the record for evaluation
            subset_key=record['subset'],
            metadata=record  # Store the full record for evaluation
        )

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        from .generation import predict
        return predict(model, sample)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
        from bfcl_eval.model_handler.utils import (
            convert_to_function_call,
            default_decode_ast_prompting,
            default_decode_execute_prompting,
        )
        from bfcl_eval.utils import is_empty_output

        from .utils import convert_format_language, convert_language

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        try:
            # NOTE: This is hardcoded dummy model since its only use is to infer underscore_to_dot
            if self.underscore_to_dot:
                dummy_model = 'gpt-4o-2024-11-20-FC'
            else:
                dummy_model = 'meta-llama/Llama-3.3-70B-Instruct-FC'

            row = task_state.metadata
            test_category = BFCL_V3_TO_V4_SUBJECT_MAPPING.get(row['test_category'], row['test_category'])

            if test_category in {'irrelevance', 'live_irrelevance', 'live_relevance'}:
                error = None
                try:
                    if self.is_fc_model:
                        decoded_tool_calls = []
                        for tool_call in row['generation'][0]:
                            name = list(tool_call.keys())[0]
                            params = tool_call[name]
                            decoded_tool_calls.append({name: params})
                    else:
                        decoded_tool_calls = default_decode_ast_prompting(
                            row['generation'][0][0], convert_format_language(row['language'])
                        )

                    # successful decode means valid function call was present
                    contains_func_call = True
                    if is_empty_output(decoded_tool_calls):
                        # Empty output is not considered as a valid function call
                        contains_func_call = False
                        error = 'Empty decoded output.'
                except Exception:
                    contains_func_call = False
                    error = f'Failed to decode with traceback: {traceback.format_exc()}'
                finally:
                    valid = contains_func_call if test_category == 'live_relevance' else not contains_func_call
                    score_result = {'valid': valid, 'error_message': error}

            elif row['multi_turn']:
                # each step might give a list of tool calls and each turn is multi-step
                # and multi-turn has generations of all the turns
                # hence in a multi-turn setting,
                # multi_turn_decoded_generations is a list of list of list of strings
                multi_turn_decoded_generations: list[list[list[str]]] = []
                for single_turn_generations in row['generation']:
                    single_turn_decoded_generations: list[list[str]] = []
                    for generation in single_turn_generations:
                        try:
                            if self.is_fc_model:
                                tool_calls = convert_to_function_call(generation)
                            else:
                                tool_calls = default_decode_execute_prompting(generation)

                            single_turn_decoded_generations.append(tool_calls)
                        except Exception:
                            single_turn_decoded_generations.append([generation])

                    multi_turn_decoded_generations.append(single_turn_decoded_generations)

                try:
                    raw_score_result = multi_turn_checker(
                        multi_turn_decoded_generations,
                        row['ground_truth'],
                        row,
                        test_category,
                        dummy_model,
                    )
                except Exception:
                    raw_score_result = {
                        'valid': False,
                        'error_type': 'multi_turn:checker_failed',
                        'error_message': f'Failed to grade multi-turn. Traceback: {traceback.format_exc()}',
                    }

                score_result = {
                    'valid': float(raw_score_result['valid']),
                    'error_message': raw_score_result.get('error_message', ''),
                    'error_type': raw_score_result.get('error_type', ''),
                }
            else:
                try:
                    if self.is_fc_model:
                        decoded_tool_calls = []
                        for tool_call in row['generation'][0]:
                            name = list(tool_call.keys())[0]
                            params = tool_call[name]
                            decoded_tool_calls.append({name: params})
                    else:
                        decoded_tool_calls = default_decode_ast_prompting(
                            row['generation'][0][0], convert_format_language(row['language'])
                        )

                    score_result = ast_checker(
                        row['functions'],
                        decoded_tool_calls,
                        row['ground_truth'],
                        convert_language(row['language']),
                        test_category,
                        dummy_model,
                    )
                except Exception:
                    score_result = {
                        'valid': False,
                        'error_message': f'Invalid syntax. Failed to decode AST. Traceback: {traceback.format_exc()}',
                        'error_type': 'ast_decoder:decoder_failed',
                    }

            score.value = {
                'acc': float(score_result['valid']),
            }
            score.explanation = score_result.get('error_message', 'Evaluation completed')
            score.metadata = {
                'raw_score_result': score_result,
                'test_category': test_category,
                'underscore_to_dot': self.underscore_to_dot,
                'is_fc_model': self.is_fc_model
            }
            score.main_score_name = 'acc'

        except Exception:
            logger.error(f'Evaluation failed for sample: {task_state.sample_id}\n{traceback.format_exc()}')
            score.value = {'acc': 0.0}
            score.explanation = 'Evaluation failed with an unexpected error.'
            score.metadata = {'error': traceback.format_exc()}
            score.main_score_name = 'acc'
        return score

    def _on_generate_report_end(self, report: Report, output_dir, **kwargs):
        """
        Finalize the report generation process. Calculate the overall score.

        Track the number of each category.
        - step1: simple, java, javascript unweighted average as simple_ast
        - step2.1: simple_ast, multiple, parallel, parallel_multiple unweighted average as ast_non_live
        - step2.2: live_simple, live_multiple, live_parallel, live_parallel_multiple weighted average as ast_live
        - step2.3: irrelevance as hallucination_non_live
        - step2.4: live_irrelevance, live_relevance weighted average as hallucination_live
        - step2.5: multi_turn_base as multi_turn_base
        - step2.6: multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context weighted average as multi_turn_augmented
        - step3.1: ast_non_live, hallucination_non_live unweighted average as non_live
        - step3.2: ast_live, hallucination_live weighted average as live
        - step3.3: multi_turn_base, multi_turn_augmented unweighted average as multi_turn
        - step4: non_live, live, multi_turn unweighted average as overall
        Args:
            report (Report): The generated evaluation report.
            output_dir (str): The directory to save the report.

        Returns:
            None
        """  # noqa: E501
        for metric in report.metrics:
            # Collect all subsets in a dictionary for easy access
            subset_dict: Dict[str, Subset] = {}
            for category in metric.categories:
                for subset in category.subsets:
                    subset_dict[subset.name] = subset

            # Step 1: Calculate simple_ast (simple, java, javascript unweighted average)
            simple_subsets = ['simple', 'java', 'javascript']
            simple_ast = unweighted_average_from_subsets(simple_subsets, subset_dict)
            subset_dict['simple_ast'] = simple_ast

            # Step 2.1: Calculate ast_non_live
            # (simple_ast, multiple, parallel, parallel_multiple unweighted average)
            ast_non_live_subsets = ['simple_ast', 'multiple', 'parallel', 'parallel_multiple']
            ast_non_live = unweighted_average_from_subsets(ast_non_live_subsets, subset_dict)
            subset_dict['ast_non_live'] = ast_non_live

            # Step 2.2: Calculate ast_live
            # (live_simple, live_multiple, live_parallel, live_parallel_multiple weighted average)
            live_subsets = ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple']
            ast_live = weighted_average_from_subsets(live_subsets, subset_dict)
            subset_dict['ast_live'] = ast_live

            # Step 2.3: hallucination_non_live (irrelevance)
            if 'irrelevance' in subset_dict:
                subset_dict['hallucination_non_live'] = subset_dict['irrelevance']
            else:
                subset_dict['hallucination_non_live'] = Subset(name='hallucination_non_live', score=0, num=0)

            # Step 2.4: Calculate hallucination_live (live_irrelevance, live_relevance weighted average)
            hallucination_live_subsets = ['live_irrelevance', 'live_relevance']
            hallucination_live = weighted_average_from_subsets(hallucination_live_subsets, subset_dict)
            subset_dict['hallucination_live'] = hallucination_live

            # Step 2.5: multi_turn_base
            if 'multi_turn_base' not in subset_dict:
                subset_dict['multi_turn_base'] = Subset(name='multi_turn_base', score=0, num=0)

            # Step 2.6: Calculate multi_turn_augmented
            # (multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context weighted average)
            multi_turn_augmented_subsets = ['multi_turn_miss_func', 'multi_turn_miss_param', 'multi_turn_long_context']
            multi_turn_augmented = weighted_average_from_subsets(multi_turn_augmented_subsets, subset_dict)
            subset_dict['multi_turn_augmented'] = multi_turn_augmented

            # Step 3.1: Calculate non_live (ast_non_live, hallucination_non_live unweighted average)
            non_live_subsets = ['ast_non_live', 'hallucination_non_live']
            non_live = unweighted_average_from_subsets(non_live_subsets, subset_dict)
            subset_dict['non_live'] = non_live

            # Step 3.2: Calculate live (ast_live, hallucination_live weighted average)
            live_agg_subsets = ['ast_live', 'hallucination_live']
            live = weighted_average_from_subsets(live_agg_subsets, subset_dict)
            subset_dict['live'] = live

            # Step 3.3: Calculate multi_turn (multi_turn_base, multi_turn_augmented unweighted average)
            multi_turn_subsets = ['multi_turn_base', 'multi_turn_augmented']
            multi_turn = unweighted_average_from_subsets(multi_turn_subsets, subset_dict)
            subset_dict['multi_turn'] = multi_turn

            # Step 4: Calculate overall (non_live, live, multi_turn unweighted average)
            overall_subsets = ['non_live', 'live', 'multi_turn']
            overall = unweighted_average_from_subsets(overall_subsets, subset_dict)
            subset_dict['overall'] = overall

            # Add computed scores to the category
            computed_subset_names = ['non_live', 'live', 'multi_turn', 'overall']

            # Add the computed scores as new subsets in the metric
            dummy_subsets = []
            for subset_name in computed_subset_names:
                if subset_name in subset_dict:
                    subset = subset_dict[subset_name]
                    subset.name = subset_name.upper()
                    dummy_subsets.append(subset)
            dummy_category = Category(name='-', subsets=dummy_subsets)
            metric.categories.append(dummy_category)
