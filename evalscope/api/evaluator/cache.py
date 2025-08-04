import json
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from evalscope.api.dataset import Dataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage, dict_to_chat_message
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ModelOutput


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
    def from_task_state(cls, state: TaskState) -> 'ModelResult':
        return cls(
            model=state.model,
            index=state.sample_id,
            messages=state.messages,
            model_output=state.output,
        )

    def to_task_state(self, dataset: Dataset) -> TaskState:
        sample = dataset[self.index]
        if not sample:
            raise ValueError(f'Sample with index {self.index} not found in dataset')

        return TaskState(
            model=self.model,
            sample=sample,
            messages=[dict_to_chat_message(msg) for msg in self.messages],
            output=ModelOutput.model_validate(self.model_output),
            completed=True,
        )


class ReviewResult(BaseModel):
    """Class to hold the result of a review."""

    index: int
    """Index of the sample in the dataset."""
    input: Dict[str, Any]
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
