import abc
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from evalscope.api.messages import ChatMessage


@dataclass
class Sample:
    r"""Sample for an evaluation task."""

    input: Union[str, List[ChatMessage]]
    """The input to be submitted to the model."""

    choices: Optional[List[str]] = None
    """List of available answer choices (used only for multiple-choice evals)."""

    target: Union[str, List[str]] = ''
    """Ideal target output. May be a literal value or narrative text to be used by a model grader."""

    id: Optional[Union[int, str]] = None
    """Unique identifier for sample."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata associated with the sample."""

    sandbox: Optional[str] = None
    """Sandbox environment type and optional config file."""

    files: Optional[dict[str, str]] = None
    """Files that go along with the sample (copied to SandboxEnvironment)"""

    setup: Optional[str] = None
    """Setup script to run for sample (run within default SandboxEnvironment)."""


@dataclass
class FieldSpec:
    r"""Specification for mapping data source fields to sample fields."""

    input: str = field(default='input')
    """Name of the field containing the sample input."""

    target: str = field(default='target')
    """Name of the field containing the sample target."""

    choices: str = field(default='choices')
    """Name of field containing the list of answer choices."""

    id: str = field(default='id')
    """ Unique identifier for the sample."""

    metadata: Optional[List[str]] = field(default=None)
    """List of additional field names that should be read as metadata."""

    sandbox: str = field(default='sandbox')
    """Sandbox type along with optional config file."""

    files: str = field(default='files')
    """Files that go along wtih the sample."""

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
    def __getitem__(self, index: Union[int, slice]) -> Union[Sample, 'Dataset']:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def filter(self, predicate: Callable[[Sample], bool], name: Optional[str] = None) -> 'Dataset':
        """Filter the dataset using a predicate.

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

    def _remap_target(self, target: Union[str, List[str]], position_map: dict[int, str]) -> Union[str, List[str]]:
        if isinstance(target, list):
            return [position_map[answer_index(t)] for t in target]
        else:
            return position_map[answer_index(target)]

    def filter(self, predicate: Callable[[Sample], bool], name: Optional[str] = None) -> 'MemoryDataset':
        return MemoryDataset(
            name=name or self.name,
            location=self.location,
            samples=[sample for sample in self if predicate(sample)],
            shuffled=self.shuffled,
        )


def answer_character(index: int) -> str:
    r"""
    Helper to go from array index to char, for example:

        0 -> 'A', 1 -> 'B', etc
    """
    if index < 26:
        return chr(ord('A') + index)
    else:
        return str(index - 25)


def answer_index(char: str) -> int:
    r"""
    Helper to go from char to array index, for example:

        'A' -> 0, 'B' -> 1, etc
    """
    if char.isalpha() or char == ',' or char == ' ':
        return ord(char.upper()) - ord('A')
    elif char.isnumeric():
        return 25 + int(char)
    else:
        raise ValueError(f'Unepxected multiple choice answer: {char} (must be a letter or number)')


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

    def items(self):
        return self.datasets.items()

    def keys(self):
        return self.datasets.keys()

    def values(self):
        return self.datasets.values()

    def __len__(self) -> int:
        return len(self.datasets)

    @classmethod
    def from_dataset(cls, dataset: Dataset, subset_key: str = None, format: str = '{}') -> 'DatasetDict':
        """
        Create a DatasetDict from a single Dataset.

        Args:
            dataset (Dataset): The dataset to wrap in a DatasetDict.

        Returns:
            DatasetDict: A new DatasetDict containing the provided dataset.
        """
        if subset_key:
            data_dict = defaultdict(list)
            dataset_dict = defaultdict(list)
            for sample in dataset:
                key = sample.metadata.get(subset_key, 'default')
                data_dict[format.format(key)].append(sample)
            for key, samples in data_dict.items():
                dataset_dict[key] = MemoryDataset(samples, name=dataset.name)
            return cls(dataset_dict)
        return cls({dataset.name or 'default': dataset})
