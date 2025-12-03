# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.dataset.loader import LocalDataLoader
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

# Import local parser functions
from evalscope.benchmarks.hmmt25.parser import check_answers, extract_answer
from evalscope.benchmarks.hmmt25.utils import WarningType, parse_answer

logger = get_logger()

PROMPT_TEMPLATE = """Please reason step by step, and put your final answer within \\boxed{{}}.

{problem}"""


@register_benchmark(
    BenchmarkMeta(
        name='hmmt25',
        pretty_name='HMMT February 2025',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'HMMT (Harvard-MIT Mathematics Tournament) February 2025 is a final-answer math competition with 30 problems. '
        'Models are evaluated by extracting answers from \\boxed{} notation and comparing against gold answers. '
        '[Homepage](https://www.hmmt.org/)',
        dataset_id='datasets/HMMT2025',  # Local dataset path
        subset_list=['train'],
        metric_list=['accuracy'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # Will look for train.json
        prompt_template=PROMPT_TEMPLATE,
    )
)
class HMMT25Adapter(DefaultDataAdapter):
    """
    Data adapter for HMMT February 2025 benchmark.

    This adapter handles:
    - Loading math problems from JSON dataset
    - Extracting answers from \\boxed{} notation
    - Grading answers using SymPy normalization and comparison
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_disk(self, use_local_loader: bool = True) -> Tuple[DatasetDict, Optional[DatasetDict]]:
        """
        Override to handle JSON file loading (not JSONL).
        
        HMMT25 dataset is stored as train.json (JSON array format),
        not JSONL format, so we need custom loading logic.
        """
        test_dataset = None
        fewshot_dataset = None
        
        # Load JSON file directly
        dataset_path = Path(self.dataset_id)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Look for train.json file
        json_file = dataset_path / 'train.json'
        if not json_file.exists():
            # Fallback: try to use LocalDataLoader for other formats
            logger.warning(f"train.json not found at {json_file}, trying LocalDataLoader...")
            test_load_func = partial(self.load_subset, data_loader=LocalDataLoader)
            test_dataset = self.load_subsets(test_load_func)
            return test_dataset, fewshot_dataset
        
        # Load JSON array file
        logger.info(f"Loading HMMT25 dataset from {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        if not isinstance(data_list, list):
            raise ValueError(f"Expected JSON array, got {type(data_list)}")

        # Apply limit if specified
        if self.limit is not None:
            if isinstance(self.limit, int):
                # Integer limit: take first N samples
                data_list = data_list[:self.limit]
            elif isinstance(self.limit, float) and 0 < self.limit < 1:
                # Float limit: take fraction of samples
                import math
                num_samples = math.ceil(len(data_list) * self.limit)
                data_list = data_list[:num_samples]
            logger.info(f"Applied limit, using {len(data_list)} samples")

        # Convert to samples
        samples = []
        for record in data_list:
            sample = self.record_to_sample(record)
            samples.append(sample)
        
        # Create DatasetDict
        from evalscope.api.dataset import MemoryDataset
        memory_dataset = MemoryDataset(
            samples=samples,
            name='hmmt25_train',
            location=str(json_file),
        )
        
        # Assign IDs to samples (reindex)
        memory_dataset.reindex()
        
        test_dataset = DatasetDict({'train': memory_dataset})
        
        return test_dataset, fewshot_dataset

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record: Dictionary containing 'problem_idx', 'problem', 'answer', 'problem_type'

        Returns:
            Sample object with problem text as input (will be formatted later)
        """
        problem = record.get('problem', '').strip()
        answer = record.get('answer', '').strip()
        problem_idx = record.get('problem_idx', 0)
        problem_type = record.get('problem_type', [])

        # Detect list answers (contains comma)
        is_list_answer = ',' in answer

        # Store raw problem text - formatting will be done in format_prompt_template
        return Sample(
            input=problem,  # Store raw problem, not formatted prompt
            target=answer,
            metadata={
                'problem_idx': problem_idx,
                'problem_type': problem_type,
                'gold_answer': answer,
                'is_list_answer': is_list_answer,      # New: list answer detection
                'strict_parsing': False,                # New: use lenient parsing
            }
        )

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Override to use 'problem' instead of 'question' in template formatting.

        Args:
            sample: The sample object containing the problem text

        Returns:
            str: The formatted prompt ready for model input
        """
        return self.prompt_template.format(problem=sample.input)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """
        Extract the final answer from model output.

        This function:
        1. Finds the last \\boxed{} or \\fbox{} in the text
        2. Removes inner boxed commands (handles nested boxes)
        3. Returns the extracted answer

        Args:
            prediction: Model's raw output text
            task_state: Current evaluation state

        Returns:
            Extracted answer string, or empty string if extraction fails
        """
        try:
            # Get metadata
            list_answer = task_state.metadata.get('is_list_answer', False)
            strict_parsing = task_state.metadata.get('strict_parsing', False)

            # Call enhanced extract_answer with list support
            extracted, warning = extract_answer(
                prediction,
                strict=strict_parsing,
                list_answer=list_answer
            )

            # Log warning if not NONE
            if warning != WarningType.NONE:
                logger.warning(f'Extraction warning: {warning.name}')
                task_state.metadata['extraction_warning'] = warning.value

            if extracted:
                logger.debug(f'Extracted answer: {extracted}, Warning: {warning.name}')
                return extracted
            else:
                logger.warning(f'Failed to extract answer from text. Warning: {warning.name}')
                return ''

        except Exception as e:
            logger.error(f'Error extracting answer: {e}')
            return ''

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores using HMMT25 grading algorithm.

        This function:
        1. Extracts the model's answer from the prediction
        2. Normalizes both model and gold answers using SymPy
        3. Compares using check_answers function
        4. Returns 1.0 (correct) or 0.0 (incorrect)

        Args:
            original_prediction: Raw model output
            filtered_prediction: Extracted answer from model output
            reference: Gold standard answer
            task_state: Current evaluation state

        Returns:
            Score object with accuracy metric
        """
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        try:
            # Get gold answer from metadata
            gold_answer = task_state.metadata.get('gold_answer', reference)

            if not gold_answer:
                logger.warning('No gold answer found')
                score.value = {'accuracy': 0.0}
                score.main_score_name = 'accuracy'
                score.explanation = 'No gold answer available'
                return score

            # If no answer was extracted, score is 0
            if not filtered_prediction:
                logger.warning('No answer extracted from model output')
                score.value = {'accuracy': 0.0}
                score.main_score_name = 'accuracy'
                score.explanation = 'Failed to extract answer from model output'
                return score

            # Normalize both answers
            try:
                is_list_answer = task_state.metadata.get('is_list_answer', False)

                # Parse with list_answer support
                gold_parsed, gold_warning = parse_answer(str(gold_answer), list_answer=is_list_answer)
                model_parsed, model_warning = parse_answer(filtered_prediction, list_answer=is_list_answer)
            except Exception as e:
                logger.error(f'Error parsing answers: {e}')
                score.value = {'accuracy': 0.0}
                score.main_score_name = 'accuracy'
                score.explanation = f'Parse error: {str(e)}'
                return score

            # Compare answers
            is_correct = check_answers(model_parsed, gold_parsed)
            accuracy = 1.0 if is_correct else 0.0

            score.value = {'accuracy': accuracy}
            score.main_score_name = 'accuracy'
            score.explanation = f'Model: {filtered_prediction}, Gold: {gold_answer}, Correct: {is_correct}'

            # Add warnings to explanation if present
            if model_warning != WarningType.NONE:
                score.explanation += f' | Model Warning: {model_warning.name}'
            if gold_warning != WarningType.NONE:
                score.explanation += f' | Gold Warning: {gold_warning.name}'

            logger.debug(f'Model answer: {filtered_prediction}')
            logger.debug(f'Gold answer: {gold_answer}')
            logger.debug(f'Accuracy: {accuracy}')

        except Exception as e:
            logger.error(f'Error calculating HMMT25 score: {e}')
            import traceback
            logger.debug(traceback.format_exc())
            score.value = {'accuracy': 0.0}
            score.main_score_name = 'accuracy'
            score.explanation = f'Evaluation error: {str(e)}'

        return score
