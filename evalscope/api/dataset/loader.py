from abc import ABC, abstractmethod

from .dataset import Dataset, Sample


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    """

    @abstractmethod
    def load(self) -> Dataset:
        """
        Load data from the source.
        """
        pass

    @abstractmethod
    def get_sample_fields(self):
        """
        Get the fields of a sample.
        """
        pass
