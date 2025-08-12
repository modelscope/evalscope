# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.filter import FilterEnsemble, build_filter_ensemble
from evalscope.api.metric import AggScore, SampleScore
from evalscope.api.mixin import LLMJudgeMixin
from evalscope.api.model import Model
from evalscope.report import Report
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.config import TaskConfig

logger = get_logger()


class DataAdapter(LLMJudgeMixin, ABC):
    """
    Data Adapter for the benchmark.
    """

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: Optional['TaskConfig'] = None):
        self._benchmark_meta = benchmark_meta
        self._task_config = task_config
        super().__init__(task_config=task_config)

        self.reformat_subset = False
        """Whether to reformat the subset data with subset key"""

        self.split_as_subset = False
        """Whether to use the split name as the dataset subsets"""
        
        self.shuffle_choices = False
        """Whether to shuffle the choices in the dataset"""

        self.category_map = {}
        """Category map for the benchmark"""

        # dataset
        self.test_dataset: Optional[DatasetDict] = None
        self.fewshot_dataset: Optional[DatasetDict] = None

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
    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        pass

    @abstractmethod
    def generate_report(self, scores: Dict[str, List[AggScore]], model_name: str, output_dir: str, **kwargs) -> Report:
        """
        Generate a report based on the evaluation results.
        """
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
    def model_adapter(self) -> Optional[str]:
        """
        Return the model adapter of the benchmark.
        """
        return self._benchmark_meta.model_adapter

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
    def subset_list(self) -> List[str]:
        """
        Return the subset list of the benchmark.
        """
        return self._benchmark_meta.subset_list

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

    @property
    def few_shot_num(self) -> int:
        """
        Return the few shot number of the benchmark.
        """
        return self._benchmark_meta.few_shot_num

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

    @property
    def eval_split(self) -> Optional[str]:
        """
        Return the eval split of the benchmark.
        """
        return self._benchmark_meta.eval_split

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
        return self._benchmark_meta.extra_params

    @property
    def seed(self) -> Optional[int]:
        """
        Return the seed for the benchmark.
        """
        return self._task_config.seed