# Copyright (c) Alibaba, Inc. and its affiliates.

import contextlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.filter import FilterEnsemble, build_filter_ensemble
from evalscope.api.metric import AggScore, SampleScore
from evalscope.api.mixin import LLMJudgeMixin, SandboxMixin
from evalscope.api.model import Model
from evalscope.report import Report
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.config import TaskConfig

logger = get_logger()


class DataAdapter(LLMJudgeMixin, SandboxMixin, ABC):
    """
    Data Adapter for the benchmark.
    """

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: Optional['TaskConfig'] = None):
        self._benchmark_meta = benchmark_meta
        self._task_config = task_config

        super().__init__(benchmark_meta=benchmark_meta, task_config=task_config)

        self.reformat_subset = False
        """Whether to reformat the subset data with subset key"""

        self.split_as_subset = False
        """Whether to use the split name as the dataset subsets"""

        self.shuffle_choices = False
        """Whether to shuffle the choices in the dataset"""

        self.use_batch_scoring = False
        """Whether to use batch scoring for metrics that support it, need to be enabled in the benchmark as well"""

        self.save_metadata = True
        """Whether to save metadata in the review result"""

        self.add_aggregation_name = True
        """Whether to add aggregation name in the report"""

        self.add_overall_metric = True
        """Whether to add overall metric in the report"""

        self.category_map = {}
        """Category map for the benchmark"""

        self.current_subset_name = ''
        """Subset name when loading datasets"""

        # dataset
        self.test_dataset: Optional[DatasetDict] = None
        """Dataset to be evaluated"""

        self.fewshot_dataset: Optional[DatasetDict] = None
        """Dataset for few-shot evaluation"""

        # filters
        self._filter_ensemble: Optional[OrderedDict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the benchmark metadata to a dictionary."""
        return self._benchmark_meta.to_string_dict()

    @abstractmethod
    def load_dataset(self) -> DatasetDict:
        pass

    @abstractmethod
    def run_inference(self, model: Model, sample: Sample, output_dir: str, **kwargs) -> TaskState:
        pass

    @abstractmethod
    def calculate_metrics(self, task_state: TaskState) -> SampleScore:
        pass

    @abstractmethod
    def batch_calculate_metrics(self, task_states: List[TaskState],
                                sample_scores: List[SampleScore]) -> List[SampleScore]:
        """Batch calculate metrics for a list of task states. Need to update sample_scores in place."""
        pass

    @abstractmethod
    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        pass

    @abstractmethod
    def generate_report(self, scores: Dict[str, List[AggScore]], model_name: str, output_dir: str, **kwargs) -> Report:
        """
        Generate a report based on the evaluation results.
        """
        pass

    @abstractmethod
    def finalize(self, *args, **kwargs) -> None:
        """Finalize the evaluation process."""
        pass

    @property
    def name(self) -> str:
        """
        Return the unique name of the benchmark.
        """
        return self._benchmark_meta.name

    @property
    def dataset_id(self) -> str:
        """
        Return the dataset ID or path to the benchmark.
        """
        return self._benchmark_meta.dataset_id

    @property
    def output_types(self) -> Optional[List[str]]:
        """
        Return the output types of the benchmark.
        """
        return self._benchmark_meta.output_types

    @property
    def limit(self) -> Optional[Union[int, float]]:
        """
        Return the limit for the benchmark.
        """
        return self._task_config.limit

    @property
    def repeats(self) -> int:
        """
        Return the number of repeats for each sample in the benchmark.
        """
        return self._task_config.repeats

    @property
    def dataset_hub(self) -> str:
        """
        Return the dataset hub type for the benchmark.
        """
        return self._task_config.dataset_hub

    @dataset_hub.setter
    def dataset_hub(self, value: str):
        """
        Set the dataset hub type for the benchmark.
        """
        self._task_config.dataset_hub = value

    @property
    def eval_type(self) -> str:
        """
        Return the evaluation type for the benchmark.
        """
        return self._task_config.eval_type

    @property
    def subset_list(self) -> List[str]:
        """
        Return the subset list of the benchmark.
        """
        return self._benchmark_meta.subset_list

    @subset_list.setter
    def subset_list(self, value: List[str]):
        """
        Set the subset list of the benchmark.
        """
        self._benchmark_meta.subset_list = value

    @property
    def metric_list(self) -> List[Union[str, Dict[str, Any]]]:
        """
        Return the metric list of the benchmark.
        """
        return self._benchmark_meta.metric_list

    @property
    def default_subset(self) -> str:
        """
        Return the default subset of the benchmark.
        """
        return self._benchmark_meta.default_subset

    @default_subset.setter
    def default_subset(self, value: str):
        """
        Set the default subset of the benchmark.
        """
        self._benchmark_meta.default_subset = value

    @property
    def few_shot_num(self) -> int:
        """
        Return the few shot number of the benchmark.
        """
        return self._benchmark_meta.few_shot_num

    @few_shot_num.setter
    def few_shot_num(self, value: int):
        """
        Set the few shot number of the benchmark.
        """
        self._benchmark_meta.few_shot_num = value

    @property
    def few_shot_random(self) -> bool:
        """
        Return whether few shot is random for the benchmark.
        """
        return self._benchmark_meta.few_shot_random

    @property
    def train_split(self) -> Optional[str]:
        """
        Return the train split of the benchmark.
        """
        return self._benchmark_meta.train_split

    @train_split.setter
    def train_split(self, value: str):
        """
        Set the train split of the benchmark.
        """
        self._benchmark_meta.train_split = value

    @property
    def eval_split(self) -> Optional[str]:
        """
        Return the eval split of the benchmark.
        """
        return self._benchmark_meta.eval_split

    @eval_split.setter
    def eval_split(self, value: str):
        """
        Set the eval split of the benchmark.
        """
        self._benchmark_meta.eval_split = value

    @property
    def prompt_template(self) -> Optional[str]:
        """
        Return the prompt template of the benchmark.
        """
        return self._benchmark_meta.prompt_template

    @prompt_template.setter
    def prompt_template(self, value: str):
        """
        Set the prompt template of the benchmark.
        """
        self._benchmark_meta.prompt_template = value

    @property
    def system_prompt(self) -> Optional[str]:
        """
        Return the system prompt of the benchmark.
        """
        return self._benchmark_meta.system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        """
        Set the system prompt of the benchmark.
        """
        self._benchmark_meta.system_prompt = value

    @property
    def query_template(self) -> Optional[str]:
        """
        Return the query template of the benchmark.
        """
        return self._benchmark_meta.query_template

    @property
    def few_shot_prompt_template(self) -> Optional[str]:
        """
        Return the few-shot prompt template of the benchmark.
        """
        return self._benchmark_meta.few_shot_prompt_template

    @property
    def pretty_name(self) -> Optional[str]:
        """
        Return the pretty name of the benchmark.
        """
        return self._benchmark_meta.pretty_name

    @property
    def description(self) -> Optional[str]:
        """
        Return the description of the benchmark.
        """
        return self._benchmark_meta.description

    @property
    def tags(self) -> Optional[List[str]]:
        """
        Return the tags of the benchmark.
        """
        return self._benchmark_meta.tags

    @property
    def filters(self) -> Optional[OrderedDict]:
        """
        Return the filters of the benchmark.
        """
        return self._benchmark_meta.filters

    @property
    def filter_ensemble(self) -> Optional[FilterEnsemble]:
        """
        Return the filter ensemble of the benchmark.
        """
        if self._filter_ensemble is None:
            if self.filters:
                self._filter_ensemble = build_filter_ensemble(filters=self.filters)
        return self._filter_ensemble

    @property
    def aggregation(self) -> str:
        """
        Return the aggregation function for the metrics.
        """
        return self._benchmark_meta.aggregation

    @property
    def extra_params(self) -> Optional[Dict]:
        """
        Return the extra parameters of the benchmark.
        """
        return self._benchmark_meta.get_extra_params()

    @property
    def seed(self) -> Optional[int]:
        """
        Return the seed for the benchmark.
        """
        return self._task_config.seed

    @property
    def dataset_dir(self) -> str:
        """
        Return the dataset directory for the benchmark.
        """
        return self._task_config.dataset_dir

    @property
    def shuffle(self) -> bool:
        """
        Return whether to shuffle the dataset before evaluation.
        """
        return self._benchmark_meta.shuffle

    @shuffle.setter
    def shuffle(self, value: bool):
        """
        Set whether to shuffle the dataset before evaluation.
        """
        self._benchmark_meta.shuffle = value

    @property
    def shuffle_choices(self) -> bool:
        """
        Return whether to shuffle the choices in multiple-choice datasets.
        """
        return self._benchmark_meta.shuffle_choices

    @shuffle_choices.setter
    def shuffle_choices(self, value: bool):
        """
        Set whether to shuffle the choices in multiple-choice datasets.
        """
        self._benchmark_meta.shuffle_choices = value

    @property
    def force_redownload(self) -> bool:
        """
        Return whether to force redownload the dataset from remote source.
        """
        return self._benchmark_meta.force_redownload

    @force_redownload.setter
    def force_redownload(self, value: bool):
        """
        Set whether to force redownload the dataset from remote source.
        """
        self._benchmark_meta.force_redownload = value

    @property
    def review_timeout(self) -> Optional[float]:
        """
        Return the timeout for the review process.
        """
        return self._benchmark_meta.review_timeout

    @review_timeout.setter
    def review_timeout(self, value: float):
        """
        Set the timeout for the review process.
        """
        self._benchmark_meta.review_timeout = value

    @property
    def sandbox_config(self) -> Dict[str, Any]:
        """
        Return the sandbox configuration for the benchmark.
        """
        return self._benchmark_meta.sandbox_config

    @sandbox_config.setter
    def sandbox_config(self, value: Dict[str, Any]):
        """
        Set the sandbox configuration for the benchmark.
        """
        self._benchmark_meta.sandbox_config = value

    @contextlib.contextmanager
    def _temporary_attribute(self, attr_name: str, new_value):
        """
        Set a temporary value for an attribute and restore the original value after the context block.

        Args:
            attr_name: The name of the attribute to temporarily set.
            new_value: The new value to set for the attribute.
        """
        had_attr = hasattr(self, attr_name)
        original_value = getattr(self, attr_name, None) if had_attr else None

        setattr(self, attr_name, new_value)
        try:
            yield
        finally:
            if had_attr:
                setattr(self, attr_name, original_value)
            else:
                delattr(self, attr_name)
