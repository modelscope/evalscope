import importlib
import json
import re
import traceback
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
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


@register_benchmark(
    BenchmarkMeta(
        name='bfcl_v3',
        pretty_name='BFCL-v3',
        tags=[Tags.FUNCTION_CALLING],
        description='Berkeley Function Calling Leaderboard (BFCL), the **first comprehensive '
        'and executable function call evaluation** '
        'dedicated to assessing Large Language Models\' (LLMs) ability to invoke '
        'functions. Unlike previous evaluations, '
        'BFCL accounts for various forms of function calls, diverse scenarios, and executability. '
        'Need to run `pip install bfcl-eval==2025.6.16` before evaluating. '
        '[Usage Example](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)',
        dataset_id='AI-ModelScope/bfcl_v3',
        subset_list=list(SUBJECT_MAPPING.keys()),
        metric_list=['acc'],
        eval_split='train',
        extra_params={
            'underscore_to_dot': True,
            'is_fc_model': True,
        }
    )
)
class BFCLAdapter(DefaultDataAdapter):
    """
    BFCL adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        spec = importlib.util.find_spec('bfcl_eval')
        if spec is None:
            raise ImportError(
                '`bfcl_eval` not found, please install it with `pip install bfcl-eval==2025.6.16` before evaluating.'
            )

        self.category_map = SUBJECT_MAPPING
        self.reformat_subset = True

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
            input=[ChatMessageUser(content='')],
            target='',  # Will use the record for evaluation
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
            test_category = re.sub(r'_[0-9_-]+$', '', row['id'])

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
                        decoded_tool_calls = default_decode_ast_prompting(row['generation'][0][0], row['language'])

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
                        decoded_tool_calls = default_decode_ast_prompting(row['generation'][0][0], row['language'])

                    score_result = ast_checker(
                        row['functions'],
                        decoded_tool_calls,
                        row['ground_truth'],
                        row['language'],
                        row['test_category'],
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
