import os
from collections import defaultdict
from typing import Any, Callable, Dict

from evalscope.api.dataset import Dataset, DatasetDict, RemoteDataLoader, Sample
from .benchmark import DataAdapter


class DefaultDataAdapter(DataAdapter):
    """
    Default Data Adapter for the benchmark.
    This class can be extended to implement specific data loading and processing logic.
    """

    def load_dataset(self) -> DatasetDict:
        if os.path.exists(self.dataset_id):
            # Load dataset from local path
            pass
        else:
            # Load dataset from remote source
            self.test_dataset = self.load_subsets(self.load_subset)

            if self.few_shot_num > 0 and self.train_split:
                self.fewshot_dataset = self.load_subsets(self.load_fewshot_subset)

        return self.test_dataset

    def load_subsets(self, load_func: Callable[[str], Dataset]) -> DatasetDict:

        if self.reformat_subset:
            subset_data = load_func(self.default_subset)
            dataset_dict = DatasetDict.from_dataset(subset_data, **self.reformat_subset)
        else:
            subset_dict = defaultdict()
            for subset in self.subset_list:
                subset_data = load_func(subset)
                subset_dict[subset] = subset_data
            dataset_dict = DatasetDict(subset_dict)
        return dataset_dict

    def load_subset(self, subset: str) -> Dataset:
        """
        Load a specific subset of the dataset.
        """
        split = subset if self.split_as_subset else self.eval_split
        subset_name = self.default_subset if self.split_as_subset else subset

        loader = RemoteDataLoader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset=subset_name,
            sample_fields=self.record_to_sample,
            limit=self._task_config.limit,
            data_source=self._task_config.dataset_hub,
        )
        return loader.load()

    def load_fewshot_subset(self, subset: str) -> Dataset:
        """
        Load a few-shot subset of the dataset.
        """
        split = subset if self.split_as_subset else self.train_split
        subset_name = self.default_subset if self.split_as_subset else subset

        loader = RemoteDataLoader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset=subset_name,
            sample_fields=self.record_to_sample,
            limit=self.few_shot_num,
            shuffle=self.few_shot_random,
            data_source=self._task_config.dataset_hub,
        )
        return loader.load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        raise NotImplementedError('This method should be implemented in subclasses')

    def sample_to_fewshot(self, sample: Sample) -> str:
        raise NotImplementedError('This method should be implemented in subclasses')

    def generate_prompts(self):
        # Implement prompt generation logic here
        pass

    def run_inference(self):
        # Implement inference logic here
        pass

    def evaluate(self):
        # Implement evaluation logic here
        pass

    def generate_report(self):
        # Implement report generation logic here
        pass
