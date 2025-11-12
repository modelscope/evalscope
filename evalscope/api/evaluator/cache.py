import copy
import os
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.api.dataset import Dataset
from evalscope.api.messages import ChatMessage
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.constants import DumpMode
from evalscope.utils.io_utils import OutputsStructure, dump_jsonl_data, jsonl_to_list
from evalscope.utils.logger import get_logger
from .state import TaskState

logger = get_logger()


class CacheManager:
    """
    Manage model results and review results for evaluation caching.

    This class handles the caching mechanism for evaluation results, allowing
    the system to resume evaluations from previously computed results and
    avoid redundant computations.
    """

    def __init__(self, outputs: OutputsStructure, model_name: str, benchmark_name: str):
        """
        Initialize the cache manager.

        Args:
            outputs: Output directory structure for storing cache files
            model_name: Name of the model being evaluated
            benchmark_name: Name of the benchmark being used
        """
        self.outputs = outputs
        self.model_name = model_name
        self.benchmark_name = benchmark_name

    def filter_prediction_cache(self, subset: str, dataset: Dataset) -> Tuple[List[TaskState], Dataset]:
        """
        Load cached prediction results and filter them from the dataset.

        This method checks for existing prediction cache files and loads any
        previously computed results. It then filters these samples from the
        input dataset to avoid recomputation.

        Args:
            subset: Name of the dataset subset
            dataset: The dataset to filter

        Returns:
            Tuple of (cached task states, filtered dataset with remaining samples)
        """
        cache_file = self.get_prediction_cache_path(subset)
        if not os.path.exists(cache_file):
            # No cache file exists, return empty cache and full dataset
            return [], dataset

        cached_task_states = []
        cached_sample_ids = set()
        cache_items = jsonl_to_list(cache_file)

        # Process each cached item
        for cache_item in cache_items:
            # Deserialize the cached model result
            cached_model_result = ModelResult.model_validate(cache_item)
            # Convert to task state for further processing
            cached_state = cached_model_result.to_task_state(dataset=dataset)

            if cached_state is None:
                continue
            cached_task_states.append(cached_state)
            cached_sample_ids.add(cached_state.sample_id)

        # Remove cached samples from the dataset to avoid reprocessing
        filtered_dataset = dataset.filter(lambda sample: sample.id not in cached_sample_ids)

        logger.info(
            f'Reusing predictions from {cache_file}, got {len(cached_task_states)} predictions, '
            f'remaining {len(filtered_dataset)} samples'
        )
        return cached_task_states, filtered_dataset

    def get_prediction_cache_path(self, subset: str) -> str:
        """
        Get the file path for prediction cache storage.

        Args:
            subset: Name of the dataset subset

        Returns:
            Path to the prediction cache file
        """
        file_path = os.path.join(self.outputs.predictions_dir, self.model_name, f'{self.benchmark_name}_{subset}.jsonl')
        # Ensure the directory exists
        if self.outputs.is_make:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def save_prediction_cache(self, subset: str, task_state: TaskState, save_metadata: bool = True) -> 'ModelResult':
        """
        Save a prediction result to the cache.

        Args:
            subset: Name of the dataset subset
            task_state: The task state containing prediction results

        Returns:
            The saved model result object
        """
        cache_file = self.get_prediction_cache_path(subset)
        # Convert task state to serializable model result
        model_result = ModelResult.from_task_state(task_state, save_metadata)
        # Serialize to dictionary
        model_result_dict = model_result.model_dump()
        # Append to JSONL cache file
        dump_jsonl_data(data_list=model_result_dict, jsonl_file=cache_file, dump_mode=DumpMode.APPEND)
        return model_result

    def filter_review_cache(self, subset: str,
                            task_states: List[TaskState]) -> Tuple[List[SampleScore], List[TaskState]]:
        """
        Load cached review results and filter corresponding task states.

        This method loads previously computed review scores and removes
        the corresponding task states from further review processing.

        Args:
            subset: Name of the dataset subset
            task_states: List of task states to potentially review

        Returns:
            Tuple of (cached sample scores, filtered task states for remaining reviews)
        """
        cache_file = self.get_review_cache_path(subset)
        if not os.path.exists(cache_file):
            # No review cache exists, return empty scores and all task states
            return [], task_states

        cached_sample_scores: List[SampleScore] = []
        cache_items = jsonl_to_list(cache_file)

        # Process each cached review result
        for cache_item in cache_items:
            # Deserialize the cached review result
            cached_review_result = ReviewResult.model_validate(cache_item)
            cached_sample_scores.append(cached_review_result.to_sample_score())

        # Filter out task states that already have review scores
        cached_sample_ids = {review.sample_id for review in cached_sample_scores}
        filtered_task_states = [state for state in task_states if state.sample_id not in cached_sample_ids]

        logger.info(f'Reusing reviews from {cache_file}, got {len(cached_sample_scores)} reviews')
        return cached_sample_scores, filtered_task_states

    def get_review_cache_path(self, subset: str) -> str:
        """
        Get the file path for review cache storage.

        Args:
            subset: Name of the dataset subset

        Returns:
            Path to the review cache file
        """
        file_path = os.path.join(self.outputs.reviews_dir, self.model_name, f'{self.benchmark_name}_{subset}.jsonl')
        # Ensure the directory exists
        if self.outputs.is_make:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def delete_review_cache(self, subset: str):
        """Delete the review cache for a specific subset. If the cache exists, it will be removed."""
        file_path = self.get_review_cache_path(subset)
        if os.path.exists(file_path):
            logger.info(f'Deleting review cache file: {file_path}')
            os.remove(file_path)

    def save_review_cache(
        self,
        subset: str,
        task_state: TaskState,
        sample_score: SampleScore,
        save_metadata: bool = True
    ) -> 'ReviewResult':
        """
        Save a review result to the cache.

        Args:
            subset: Name of the dataset subset
            task_state: The task state that was reviewed
            sample_score: The computed score for the sample

        Returns:
            The saved review result object
        """
        cache_file = self.get_review_cache_path(subset)
        # Convert score and state to serializable review result
        review_result = ReviewResult.from_score_state(sample_score, task_state, save_metadata)
        # Serialize to dictionary
        review_result_dict = review_result.model_dump()
        # Append to JSONL cache file
        dump_jsonl_data(data_list=review_result_dict, jsonl_file=cache_file, dump_mode=DumpMode.APPEND)
        return review_result

    def get_report_path(self) -> str:
        """
        Get the directory path for report storage.

        Returns:
            Path to the reports directory for this model
        """
        report_path = os.path.join(self.outputs.reports_dir, self.model_name)
        # Ensure the directory exists
        if self.outputs.is_make:
            os.makedirs(report_path, exist_ok=True)
        return report_path

    def get_report_file(self) -> str:
        """
        Get the report file path for the benchmark.

        The report file is named as '{benchmark_name}.json' and contains
        the final evaluation results for the benchmark.

        Returns:
            Full path to the benchmark report file
        """
        return os.path.join(self.get_report_path(), f'{self.benchmark_name}.json')


class ModelResult(BaseModel):
    """
    Serializable container for model prediction results.

    This class represents a single model prediction that can be cached
    and restored later to avoid recomputation.
    """

    index: int
    """Index of the sample in the dataset that was processed."""

    model: str = ''
    """Name of the model that generated this prediction."""

    model_output: Optional[ModelOutput] = None
    """The actual prediction/output generated by the model."""

    messages: List[ChatMessage] = []
    """Chat messages exchanged during evaluation (for conversational models)."""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata associated with the model result."""

    @classmethod
    def from_task_state(cls, task_state: TaskState, save_metadata: bool = True) -> 'ModelResult':
        """
        Create a ModelResult from a TaskState for caching.

        Args:
            task_state: The completed task state to serialize

        Returns:
            ModelResult object ready for caching
        """
        return cls(
            model=task_state.model,
            index=task_state.sample_id,
            messages=task_state.messages,
            model_output=task_state.output,
            metadata=task_state.metadata if save_metadata else {},
        )

    def to_task_state(self, dataset: Dataset) -> TaskState:
        """
        Restore a TaskState from cached ModelResult.

        Args:
            dataset: The dataset to retrieve the original sample from

        Returns:
            Reconstructed TaskState with cached results

        Raises:
            ValueError: If the sample index is not found in the dataset
        """
        try:
            sample = dataset[self.index]
        except IndexError:
            logger.warning(f'Sample index {self.index} not found in dataset during cache restoration.')
            return None

        # update metadata if exists
        if self.metadata:
            sample.metadata.update(self.metadata)

        return TaskState(
            model=self.model,
            sample=sample,
            messages=self.messages,
            output=ModelOutput.model_validate(self.model_output),
            completed=True,  # Mark as completed since it was cached
        )

    def pretty_print(self) -> str:
        """
        Generate a pretty-printed string representation of the model result.

        Returns:
            A string representation of the model result
        """
        return self.model_dump_json(indent=2)


class ReviewResult(BaseModel):
    """
    Serializable container for review/scoring results.

    This class represents the result of reviewing a model's prediction,
    including the computed score and relevant context.
    """

    index: int
    """Index of the sample that was reviewed."""

    input: str = ''
    """Original input from the sample (immutable reference)."""

    target: Optional[str] = None
    """Expected/target answer for the sample, if available."""

    sample_score: SampleScore
    """The computed evaluation score for this sample."""

    @classmethod
    def from_score_state(
        cls, sample_score: SampleScore, state: TaskState, save_metadata: bool = True
    ) -> 'ReviewResult':
        """
        Create a ReviewResult from a score and task state for caching.

        Args:
            sample_score: The computed score for the sample
            state: The task state containing sample information

        Returns:
            ReviewResult object ready for caching
        """
        if not save_metadata:
            sample_score = copy.deepcopy(sample_score)
            sample_score.sample_metadata = None

        return cls(
            index=state.sample_id,
            input=state.input_markdown,
            target=state.target,
            sample_score=sample_score,
        )

    def to_sample_score(self) -> SampleScore:
        """
        Extract the sample score from the cached review result.

        Returns:
            The sample score object
        """
        return self.sample_score

    def pretty_print(self) -> str:
        """
        Generate a pretty-printed string representation of the review result.

        Returns:
            A string representation of the review result
        """
        output = [
            f'Review Result for Sample {self.index}:',
            f'Target: {self.target}',
            f'Score: {self.sample_score.model_dump_json(indent=2)}',
        ]
        return '\n'.join(output)
