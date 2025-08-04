import json
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union

from evalscope.api.dataset import Dataset
from evalscope.api.messages import ChatMessage
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from .state import TaskState


class ModelResult(BaseModel):
    """Class to hold the result of a model prediction."""

    index: int
    """Index of the sample in the dataset."""

    model: str = ''
    """Model name used for the prediction."""

    model_output: Optional[ModelOutput] = None
    """The prediction made by the model."""

    messages: List[ChatMessage] = []
    """Messages exchanged during the evaluation, if applicable."""

    @classmethod
    def from_task_state(cls, task_state: TaskState) -> 'ModelResult':
        return cls(
            model=task_state.model,
            index=task_state.sample_id,
            messages=task_state.messages,
            model_output=task_state.output,
        )

    def to_task_state(self, dataset: Dataset) -> TaskState:
        sample = dataset[self.index]
        if not sample:
            raise ValueError(f'Sample with index {self.index} not found in dataset')

        return TaskState(
            model=self.model,
            sample=sample,
            messages=self.messages,
            output=ModelOutput.model_validate(self.model_output),
            completed=True,
        )


class ReviewResult(BaseModel):
    """Class to hold the result of a review."""

    index: int
    """Index of the sample in the dataset."""
    input: Union[str, List[ChatMessage]] = ''
    """Input from the sample, should be considered immutable."""
    target: Optional[str] = None
    """Target answer for the sample, if applicable."""
    sample_score: SampleScore
    """The score for the sample."""

    @classmethod
    def from_score_state(cls, sample_score: SampleScore, state: TaskState) -> 'ReviewResult':
        return cls(
            index=state.sample_id,
            input=state.input,
            taget=state.target,
            sample_score=sample_score,
        )

    def to_sample_score(self) -> SampleScore:
        return self.sample_score
