import abc
from typing import TYPE_CHECKING, List, Union

from evalscope.api.metric import SampleScore
from evalscope.report import Report
from .state import TaskState

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.model import Model
    from evalscope.config import TaskConfig
    from evalscope.utils.io_utils import OutputsStructure


class Evaluator(abc.ABC):
    """
    Abstract base class for evaluators.

    Args:
        benchmark (DataAdapter): The data adapter for the benchmark.
        model (Model): The model to evaluate.
        outputs (OutputsStructure, optional): The output structure for results.
        task_config (TaskConfig, optional): The task configuration.
    """

    def __init__(
        self,
        benchmark: 'DataAdapter',
        model: 'Model',
        outputs: 'OutputsStructure' = None,
        task_config: 'TaskConfig' = None,
    ):
        self.benchmark = benchmark
        self.model = model
        self.outputs = outputs
        self.task_config = task_config

    @abc.abstractmethod
    def eval(self, *args, **kwargs) -> Report:
        """Run the evaluation process."""
        pass

    @abc.abstractmethod
    def get_answers(self, *args, **kwargs) -> List[TaskState]:
        """Get the evaluation answers."""
        pass

    @abc.abstractmethod
    def get_reviews(self, *args, **kwargs) -> List[SampleScore]:
        """Get the review results."""
        pass

    @abc.abstractmethod
    def get_report(self, *args, **kwargs) -> Report:
        """Get the evaluation report."""
        pass

    @abc.abstractmethod
    def finalize(self, *args, **kwargs) -> None:
        """Finalize the evaluation process."""
        pass
