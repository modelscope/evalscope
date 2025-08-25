from abc import ABC
from collections import defaultdict
from typing import Any, Callable, Dict

from evalscope.api.dataset import Dataset, DatasetDict, RemoteDataLoader


class DatasetLoaderMixin:
    """
    Mixin class providing dataset loading functionality for benchmarks.

    This mixin provides common dataset loading methods that can be shared
    across different data adapters, including support for:
    - Loading multiple subsets
    - Few-shot dataset loading
    - Remote dataset loading with configuration
    """

    def load_subsets(self, load_func: Callable[[str], Dataset]) -> DatasetDict:
        """
        Load multiple subsets of the dataset using the provided loading function.

        This method handles two loading strategies:
        1. Reformat mode: Load only the default subset and reformat it
        2. Multi-subset mode: Load all subsets specified in subset_list

        Args:
            load_func (Callable[[str], Dataset]): Function to load individual subsets

        Returns:
            DatasetDict: Dictionary containing all loaded subsets
        """
        if self.reformat_subset:
            # Load only the default subset
            subset_data = load_func(self.default_subset)
            # Reformat the subset to create multiple subsets based on sample keys
            # NOTE: subset_list and limit is applied here if specified
            dataset_dict = DatasetDict.from_dataset(dataset=subset_data, subset_list=self.subset_list, limit=self.limit)
        else:
            # Load all specified subsets into separate entries
            subset_dict = defaultdict()
            for subset in self.subset_list:
                subset_data = load_func(subset)
                subset_dict[subset] = subset_data
            dataset_dict = DatasetDict(subset_dict)
        return dataset_dict

    def load_subset(self, subset: str) -> Dataset:
        """
        Load a specific subset of the dataset for evaluation.

        This method configures and executes the data loading for a single subset,
        handling both split-as-subset and traditional subset configurations.

        Args:
            subset (str): The subset identifier to load

        Returns:
            Dataset: The loaded dataset subset with processed samples
        """
        # Determine the split and subset names based on configuration
        split = subset if self.split_as_subset else self.eval_split
        subset_name = self.default_subset if self.split_as_subset else subset

        # Create and configure the remote data loader
        loader = RemoteDataLoader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset=subset_name,
            sample_fields=self.record_to_sample,  # Custom sample conversion function
            limit=self.limit if not self.reformat_subset else None,  # Limit number of samples if specified
            repeats=self._task_config.repeats,  # Number of repetitions for each sample
            data_source=self._task_config.dataset_hub,  # Data source configuration
        )
        return loader.load()

    def load_fewshot_subset(self, subset: str) -> Dataset:
        """
        Load a subset specifically for few-shot examples.

        This method loads training data to be used as demonstrations in few-shot prompting.
        It typically loads from the training split with limited samples and optional shuffling.

        Args:
            subset (str): The subset identifier to load few-shot examples from

        Returns:
            Dataset: The loaded few-shot dataset with demonstration examples
        """
        # Use training split for few-shot examples
        split = subset if self.split_as_subset else self.train_split
        subset_name = self.default_subset if self.split_as_subset else subset

        # Create loader specifically configured for few-shot sampling
        loader = RemoteDataLoader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset=subset_name,
            sample_fields=self.record_to_sample,
            limit=self.few_shot_num
            if not self.reformat_subset else None,  # Limit to specified number of few-shot examples
            shuffle=self.few_shot_random,  # Randomize selection if enabled
            data_source=self._task_config.dataset_hub,
        )
        return loader.load()
