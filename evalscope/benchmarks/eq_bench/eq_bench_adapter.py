# Copyright (c) Alibaba, Inc. and its affiliates.

"""
EQ-Bench Benchmark Adapter for EvalScope

This adapter implements the EQ-Bench benchmark for evaluating language models
on emotional intelligence tasks. It uses the official EQ-Bench scoring algorithm
from the reference implementation to ensure 100% consistency.

The scoring functions are imported from answer_validation.py, which is based on
the official EQ-Bench reference implementation located in reference/EQ-bench/.

References:
- Paper: https://arxiv.org/abs/2312.06281
- Homepage: https://eqbench.com/
- Official Implementation: reference/EQ-bench/answer_validation.py
"""

import ast
import json
import os
from functools import partial
from typing import Any, Dict, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.dataset.loader import LocalDataLoader
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

# Import official EQ-Bench evaluation functions from bundled implementation
# The answer_validation.py module is bundled with this package to ensure
# it's available both in development (pip install -e .) and production (pip install .) installations.
# This ensures 100% consistency with official EQ-Bench scoring algorithm.
from evalscope.benchmarks.eq_bench.answer_validation import (
    calculate_score_fullscale,
    parse_answers,
    validate_answer_format,
)

logger = get_logger()

PROMPT_TEMPLATE = """{question}"""


@register_benchmark(
    BenchmarkMeta(
        name='eq_bench',
        pretty_name='EQ-Bench',
        tags=[Tags.INSTRUCTION_FOLLOWING],
        description=
        'EQ-Bench is a benchmark for evaluating language models on emotional intelligence tasks. '
        'It assesses the ability to predict the likely emotional responses of characters in dialogues '
        'by rating the intensity of possible emotional responses. '
        '[Paper](https://arxiv.org/abs/2312.06281) | [Homepage](https://eqbench.com/)',
        dataset_id='cc7704/EQ-bench',  # ModelScope dataset ID, falls back to local if not found
        subset_list=['main'],
        metric_list=['eq_bench_score'],  # Official v2 full-scale scoring
        few_shot_num=0,
        train_split=None,
        eval_split='validation',  # Will look for main_validation.csv or validation.csv
        prompt_template=PROMPT_TEMPLATE,
    )
)
class EQBenchAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_remote(self) -> Tuple[DatasetDict, Optional[DatasetDict]]:
        """
        Load dataset from ModelScope and use LocalDataLoader for CSV files.
        
        This method downloads the dataset from ModelScope using dataset_snapshot_download,
        then uses LocalDataLoader to load CSV files, similar to how gsm8k loads from ModelScope.
        
        Returns:
            Tuple[DatasetDict, Optional[DatasetDict]]: The test dataset and few-shot dataset.
        """
        dataset_name_or_path = self.dataset_id
        
        # Download dataset from ModelScope
        try:
            from modelscope import dataset_snapshot_download
            logger.info(f'Loading EQ-Bench dataset from ModelScope: {dataset_name_or_path}')
            # Download dataset snapshot, allow CSV files
            dataset_path = dataset_snapshot_download(
                dataset_name_or_path, 
                allow_file_pattern='*.csv'
            )
        except Exception as e:
            raise FileNotFoundError(
                f'Could not load EQ-Bench dataset from ModelScope ({dataset_name_or_path}). Error: {e}'
            )

        # Temporarily update dataset_id to point to the downloaded path
        # Use _benchmark_meta to modify dataset_id since it's a property
        original_dataset_id = self._benchmark_meta.dataset_id
        self._benchmark_meta.dataset_id = dataset_path
        
        try:
            # Use LocalDataLoader for CSV file loading
            test_load_func = partial(self.load_subset, data_loader=LocalDataLoader)
            test_dataset = self.load_subsets(test_load_func)
            
            # Load few-shot examples if few-shot prompting is enabled
            fewshot_dataset = None
            if self._should_load_fewshot():
                fewshot_load_func = partial(self.load_fewshot_subset, data_loader=LocalDataLoader)
                fewshot_dataset = self.load_subsets(fewshot_load_func, is_fewshot=True)
        finally:
            # Restore original dataset_id
            self._benchmark_meta.dataset_id = original_dataset_id
        
        return test_dataset, fewshot_dataset

    def load_from_disk(self, use_local_loader: bool = True) -> Tuple[DatasetDict, Optional[DatasetDict]]:
        """
        Override to force use LocalDataLoader for CSV files.
        
        Args:
            use_local_loader: If True, use LocalDataLoader for CSV file loading.
        
        Returns:
            Tuple[DatasetDict, Optional[DatasetDict]]: The test dataset and few-shot dataset.
        """
        test_dataset = None
        fewshot_dataset = None
        
        # Use LocalDataLoader for CSV file loading
        test_load_func = partial(self.load_subset, data_loader=LocalDataLoader)
        test_dataset = self.load_subsets(test_load_func)
        
        # Load few-shot examples if few-shot prompting is enabled
        if self._should_load_fewshot():
            fewshot_load_func = partial(self.load_fewshot_subset, data_loader=LocalDataLoader)
            fewshot_dataset = self.load_subsets(fewshot_load_func, is_fewshot=True)
        
        return test_dataset, fewshot_dataset
    
    def format_prompt_template(self, sample: Sample) -> str:
        """
        Override to use 'prompt' field from sample input directly.
        Since the prompt field already contains the full formatted prompt,
        we can return it directly or format with the template.
        """
        # The sample.input is already the full prompt text, so we can use it directly
        if isinstance(sample.input, str):
            return sample.input
        # If it's a list of messages, convert to string
        return str(sample.input)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.
        
        Args:
            record: Dictionary containing 'prompt', 'reference_answer', 'reference_answer_fullscale'
            
        Returns:
            Sample object with prompt as input and reference_answer as target
        """
        prompt = record.get('prompt', '').strip()
        reference_answer = record.get('reference_answer', '')
        reference_answer_fullscale = record.get('reference_answer_fullscale', '')
        
        # Parse the reference answer dictionaries
        # CSV files typically use Python dict format (single quotes), not JSON
        try:
            if isinstance(reference_answer, str):
                # Remove outer quotes if present (CSV may wrap the dict string in quotes)
                ref_str = reference_answer.strip()
                if ref_str.startswith('"') and ref_str.endswith('"'):
                    ref_str = ref_str[1:-1]
                elif ref_str.startswith("'") and ref_str.endswith("'"):
                    ref_str = ref_str[1:-1]
                
                # Try ast.literal_eval first (for Python dict format)
                try:
                    ref_answer_dict = ast.literal_eval(ref_str)
                except (ValueError, SyntaxError):
                    # Fallback to JSON if literal_eval fails
                    ref_answer_dict = json.loads(ref_str)
            else:
                ref_answer_dict = reference_answer
                
            if isinstance(reference_answer_fullscale, str):
                ref_str = reference_answer_fullscale.strip()
                if ref_str.startswith('"') and ref_str.endswith('"'):
                    ref_str = ref_str[1:-1]
                elif ref_str.startswith("'") and ref_str.endswith("'"):
                    ref_str = ref_str[1:-1]
                    
                try:
                    ref_fullscale_dict = ast.literal_eval(ref_str)
                except (ValueError, SyntaxError):
                    ref_fullscale_dict = json.loads(ref_str)
            else:
                ref_fullscale_dict = reference_answer_fullscale
        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            logger.warning(f'Failed to parse reference answer: {e}')
            ref_answer_dict = {}
            ref_fullscale_dict = {}
        
        return Sample(
            input=prompt,
            target=str(reference_answer),  # Store as string for comparison
            metadata={
                'reference_answer': ref_answer_dict,
                'reference_answer_fullscale': ref_fullscale_dict,
            }
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores using official EQ-Bench v2 scoring algorithm.

        Uses sigmoid scaling for small differences (≤5) and linear scaling for large differences (>5),
        with an adjustment constant (0.7477) that makes random answers score 0.

        Returns a score in range 0-100 (internally 0-10 scaled by 10).
        """
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        try:
            # Get reference answer from metadata - prefer fullscale (v2) over standard (v1)
            ref_answer = task_state.metadata.get('reference_answer', {})
            ref_fullscale = task_state.metadata.get('reference_answer_fullscale', {})
            # Use fullscale if available (recommended v2 scoring), otherwise fall back to v1
            reference_to_use = ref_fullscale if ref_fullscale else ref_answer

            if not reference_to_use:
                logger.warning('No reference answer found in metadata')
                score.value = {'eq_bench_score': 0.0}
                score.main_score_name = 'eq_bench_score'
                score.explanation = 'No reference answer available'
                return score

            # Parse the prediction using official parser
            # This extracts {emotion_name: score} format from the model output
            first_pass_answers, _ = parse_answers(filtered_prediction, REVISE=False)

            if not first_pass_answers:
                logger.warning('Failed to parse any emotion scores from prediction')
                score.value = {'eq_bench_score': 0.0}
                score.main_score_name = 'eq_bench_score'
                score.explanation = 'Failed to parse prediction'
                return score

            # Get reference emotion names for validation
            reference_emotions = [
                reference_to_use.get(f'emotion{i}', '')
                for i in range(1, 5)
            ]

            # Validate answer format using official validator
            is_valid, error_msg = validate_answer_format(first_pass_answers, reference_emotions)

            if not is_valid:
                logger.warning(f'Invalid answer format: {error_msg}')
                logger.debug(f'Parsed answers: {first_pass_answers}')
                logger.debug(f'Expected emotions: {reference_emotions}')
                score.value = {'eq_bench_score': 0.0}
                score.main_score_name = 'eq_bench_score'
                score.explanation = f'Invalid format: {error_msg}'
                return score

            # Calculate score using official v2 full-scale scoring algorithm
            eq_score = calculate_score_fullscale(reference_to_use, first_pass_answers)

            if eq_score is None:
                logger.warning('calculate_score_fullscale returned None')
                score.value = {'eq_bench_score': 0.0}
                score.main_score_name = 'eq_bench_score'
                score.explanation = 'Scoring failed'
                return score

            # Scale from 0-10 to 0-100 for reporting
            normalized_score = eq_score * 10.0

            # Clamp to valid range [0, 100]
            normalized_score = max(0.0, min(100.0, normalized_score))

            score.value = {'eq_bench_score': normalized_score}
            score.main_score_name = 'eq_bench_score'
            score.explanation = f'EQ-Bench Score: {normalized_score:.2f}/100 (raw: {eq_score:.3f}/10)'

            logger.debug(f'Prediction: {first_pass_answers}')
            logger.debug(f'Reference: {reference_emotions}')
            logger.debug(f'Score: {normalized_score:.2f}')

        except Exception as e:
            logger.error(f'Error calculating EQ-Bench score: {e}')
            import traceback
            logger.debug(traceback.format_exc())
            score.value = {'eq_bench_score': 0.0}
            score.main_score_name = 'eq_bench_score'
            score.explanation = f'Evaluation error: {str(e)}'

        return score

