import abc
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from evalscope.api.messages import ChatMessage, messages_to_markdown
from evalscope.api.tool import ToolInfo


class Sample(BaseModel):
    """Sample for an evaluation task."""

    input: Union[str, List[ChatMessage]]
    """The input to be submitted to the model."""

    choices: Optional[List[str]] = None
    """List of available answer choices (used only for multiple-choice evals)."""

    target: Union[str, List[str]] = ''
    """Ideal target output. May be a literal value or narrative text to be used by a model grader."""

    id: Optional[int] = None
    """Unique identifier for sample."""

    group_id: Optional[int] = None
    """Identifier for the group this sample belongs to, used for grouping k repeated samples."""

    tools: Optional[List[ToolInfo]] = None
    """List of tools available to the model during inference (optional)."""

    subset_key: Optional[str] = None
    """Key for the subset this sample belongs to, used for generating subsets (optional)."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Arbitrary metadata associated with the sample."""

    sandbox: Optional[str] = None
    """Sandbox environment type and optional config file."""

    files: Optional[Dict[str, str]] = None
    """Files that go along with the sample (copied to SandboxEnvironment)"""

    setup: Optional[str] = None
    """Setup script to run for sample (run within default SandboxEnvironment)."""

    def pretty_print(self) -> str:
        """Return a pretty-printed string representation of the sample."""
        if isinstance(self.input, str):
            input_text = self.input
        else:
            input_text = messages_to_markdown(self.input, max_length=50)
        return f'Sample ID: {self.id}\nInput: {input_text}\nTarget: {self.target}'


@dataclass
class FieldSpec:
    r"""Specification for mapping data source fields to sample fields."""

    input: str = field(default='input')
    """Name of the field containing the sample input."""

    target: str = field(default='target')
    """Name of the field containing the sample target."""

    choices: str = field(default='choices')
    """Name of field containing the list of answer choices."""

    id: int = field(default=0)
    """ Unique identifier for the sample."""

    metadata: Optional[List[str]] = field(default=None)
    """List of additional field names that should be read as metadata."""

    sandbox: str = field(default='sandbox')
    """Sandbox type along with optional config file."""

    files: str = field(default='files')
    """Files that go along with the sample."""

    setup: str = field(default='setup')
    """Setup script to run for sample (run within default SandboxEnvironment)."""


class Dataset(Sequence[Sample], abc.ABC):
    r"""A sequence of Sample objects.

    Datasets provide sequential access (via conventional indexes or slicing)
    to a collection of Sample objects.
    """

    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        ...

    @property
    @abc.abstractmethod
    def location(self) -> Optional[str]:
        ...

    @property
    @abc.abstractmethod
    def shuffled(self) -> bool:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        """Return an iterator over the samples."""
        ...

    @abc.abstractmethod
    def __getitem__(self, index: Union[int, slice]) -> Union[Sample, 'Dataset']:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def filter(self, predicate: Callable[[Sample], bool], name: Optional[str] = None) -> 'Dataset':
        """Filter the dataset using a predicate. Only samples matching the predicate will be included.

        Args:
          predicate: Filtering function.
          name: Name for filtered dataset (optional).

        Returns:
          Filtered dataset.
        """
        ...

    @abc.abstractmethod
    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the order of the dataset (in place).

        Args:
           seed: Random seed for shuffling (optional).
        """
        ...

    @abc.abstractmethod
    def shuffle_choices(self, seed: Optional[int] = None) -> None:
        """Shuffle the order of the choices with each sample.

        Args:
           seed: Random seed for shuffling (optional).
        """
        ...

    @abc.abstractmethod
    def reindex(self, group_size=1):
        """Reindex the dataset samples to ensure consistent ordering.

        Args:
           group_size: Number of samples per group for setting group_id.
        """
        ...


class MemoryDataset(Dataset):
    r"""A Dataset stored in memory."""

    def __init__(
        self,
        samples: List[Sample],
        name: Optional[str] = None,
        location: Optional[str] = None,
        shuffled: bool = False,
    ) -> None:
        r"""A dataset of samples held in an in-memory list.

        Datasets provide sequential access (via conventional indexes or slicing)
        to a collection of Sample objects. The ListDataset is explicitly
        initialized with a list that is held in memory.

        Args:
            samples (List[Sample]): The list of sample objects.
            name (str | None): Optional name for dataset.
            location (str | None): Optional location for dataset.
            shuffled (bool): Was the dataset shuffled after reading.
        """
        self.samples = samples
        self._name = name
        self._location = location
        self._shuffled = shuffled

    @property
    def name(self) -> Optional[str]:
        """Dataset name."""
        return self._name

    @property
    def location(self) -> Optional[str]:
        """Dataset location."""
        return self._location

    @property
    def shuffled(self) -> bool:
        """Was the dataset shuffled."""
        return self._shuffled

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, index: Union[int, slice]) -> Union[Sample, Dataset]:
        if isinstance(index, int):
            return self.samples[index]
        else:
            return MemoryDataset(
                samples=self.samples[index],
                name=self.name,
                location=self.location,
                shuffled=self.shuffled,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def shuffle(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.Random(seed).shuffle(self.samples)
        else:
            random.shuffle(self.samples)
        self._shuffled = True

    def shuffle_choices(self, seed: Optional[int] = None) -> None:
        from evalscope.utils.multi_choices import answer_character

        rand = random.Random(seed)
        for sample in self.samples:
            if not sample.choices:
                continue
            # The original positions
            positions = list(range(len(sample.choices)))

            # Shuffle the choices
            rand.shuffle(positions)
            shuffled_choices = [sample.choices[i] for i in positions]

            # Map of original position / target letter
            position_map = {i: answer_character(new_i) for new_i, i in enumerate(positions)}

            # Update to the shuffled choices and target
            sample.choices = shuffled_choices
            sample.target = self._remap_target(sample.target, position_map=position_map)

    def _remap_target(self, target: Union[str, List[str]], position_map: Dict[int, str]) -> Union[str, List[str]]:
        from evalscope.utils.multi_choices import answer_index

        if isinstance(target, list):
            return [position_map[answer_index(t)] for t in target]
        else:
            return position_map[answer_index(target)]

    def filter(self, predicate: Callable[[Sample], bool], name: Optional[str] = None) -> 'MemoryDataset':
        return MemoryDataset(
            name=name or self.name,
            location=self.location,
            samples=[sample for sample in self.samples if predicate(sample)],
            shuffled=self.shuffled,
        )

    def reindex(self, group_size=1):
        # Reindex the dataset samples to ensure consistent ordering
        for i, sample in enumerate(self.samples):
            sample.id = i
            sample.group_id = i // group_size


class DatasetDict:
    """
    A dictionary-like container for datasets.
    """

    def __init__(self, datasets: Dict[str, Dataset]):
        self.datasets = datasets

    def __getitem__(self, key: str) -> Dataset:
        return self.datasets[key]

    def __setitem__(self, key: str, value: Dataset) -> None:
        self.datasets[key] = value

    def __delitem__(self, key: str) -> None:
        del self.datasets[key]

    def get(self, key: str, default: Optional[Dataset] = None) -> Optional[Dataset]:
        return self.datasets.get(key, default)

    def items(self):
        return self.datasets.items()

    def keys(self):
        return self.datasets.keys()

    def values(self):
        return self.datasets.values()

    def __len__(self) -> int:
        return len(self.datasets)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        subset_list: List[str],
        limit: Optional[Union[int, float]] = None,
        repeats: int = 1
    ) -> 'DatasetDict':
        """
        Create a DatasetDict from a single Dataset using subset key in the sample.

        Args:
            dataset (Dataset): The dataset to wrap in a DatasetDict.
            subset_list (List[str]): List of subset keys to include.
            limit (int | float | None): Optional limit on number of samples per subset.
                If int, limits to that many samples. If float, limits to that fraction of samples.

        Returns:
            DatasetDict: A new DatasetDict containing the provided dataset.
        """
        data_dict = defaultdict(list)
        dataset_dict = defaultdict(list)
        # init subset keys to prevent order issues
        for key in subset_list:
            data_dict[key] = []
            dataset_dict[key] = []

        # Loop through each sample in the dataset
        for sample in dataset.samples:
            subset_key = sample.subset_key or 'default'
            data_dict[subset_key].append(sample)
        # Create a MemoryDataset for each subset key
        for key, samples in data_dict.items():
            if key not in subset_list:
                continue
            # Apply limit if specified
            if limit is not None:
                if isinstance(limit, float):
                    limit = int(len(samples) * limit)
                total_limit = limit * repeats
                samples = samples[:total_limit]
            cur_dataset = MemoryDataset(samples, name=dataset.name)
            # Reindex the dataset to ensure consistent IDs and group IDs
            cur_dataset.reindex(group_size=repeats)
            dataset_dict[key] = cur_dataset
        return cls(dataset_dict)

    @classmethod
    def from_dataset_dicts(cls, dataset_dicts: List['DatasetDict']) -> 'DatasetDict':
        """
        Create a DatasetDict by merging multiple DatasetDicts.

        Args:
            dataset_dicts (List[DatasetDict]): List of DatasetDicts to merge.

        Returns:
            DatasetDict: A new DatasetDict containing the merged datasets.
        """
        merged_dict = defaultdict(list)
        for dataset_dict in dataset_dicts:
            for key, dataset in dataset_dict.items():
                merged_dict[key].extend(dataset.samples)
        # Create a MemoryDataset for each subset key
        final_dict = {}
        for key, samples in merged_dict.items():
            final_dict[key] = MemoryDataset(samples, name=key)
        return cls(final_dict)
