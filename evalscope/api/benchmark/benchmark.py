# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.config import TaskConfig
from evalscope.api.dataset import DatasetDict
from evalscope.utils.logger import get_logger

logger = get_logger()


class DataAdapter(ABC):
    """
    Data Adapter for the benchmark.
    """

    def __init__(self, benchmark_meta: 'BenchmarkMeta', task_config: Optional['TaskConfig']=None):
        self._benchmark_meta = benchmark_meta
        self._task_config = task_config

        self.reformat_subset = {}
        self.split_as_subset = False
        self.use_llm_judge = False

        # dataset
        self.test_dataset: Optional[DatasetDict] = None
        self.fewshot_dataset: Optional[DatasetDict] = None

    @abstractmethod
    def load_dataset(self) -> DatasetDict:
        pass

    @abstractmethod
    def generate_prompts(self):
        pass

    @abstractmethod
    def run_inference(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def generate_report(self):
        """
        Generate a report based on the evaluation results.
        """
        pass

    @property
    def name(self) -> str:
        """
        Return the name of the benchmark.
        """
        return self._benchmark_meta.name

    @property
    def dataset_id(self) -> str:
        """
        Return the dataset ID of the benchmark.
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
    def subset_list(self) -> List[str]:
        """
        Return the subset list of the benchmark.
        """
        return self._benchmark_meta.subset_list

    @property
    def metric_list(self) -> List[str]:
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
    def extra_params(self) -> Optional[Dict]:
        """
        Return the extra parameters of the benchmark.
        """
        return self._benchmark_meta.extra_params
