from typing import Any, Dict, List, Optional, Union

from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessage, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import ModelOutput


class TaskState:
    """
    The `TaskState` represents the internal state of the `Task` being run for a single `Sample`.

    The `TaskState` is passed to and returned from each solver during a sample's
    evaluation. It allows us to maintain the manipulated message history, the tools
    available to the model, the final output of the model, and whether the task
    is completed or has hit a limit.
    """

    def __init__(
        self,
        model: str,
        sample: Sample,
        messages: List[ChatMessage] = [],
        output: Optional[ModelOutput] = None,
        completed: bool = False,
    ) -> None:
        self._model = model
        self._sample = sample
        self._sample_id = sample.id
        self._group_id = sample.group_id
        self._input = sample.input
        self._target = sample.target
        self._metadata = sample.metadata
        self._messages: List[ChatMessage] = messages
        self._output = output if output else ModelOutput(model=str(model))
        self._completed = completed

    @property
    def model(self) -> str:
        """Name of model being evaluated."""
        return self._model

    @property
    def sample_id(self) -> Union[int, str]:
        """Unique id for sample."""
        return self._sample_id

    @property
    def group_id(self) -> Union[int, str]:
        """Group id for sample."""
        return self._group_id

    @property
    def input(self) -> Union[str, List[ChatMessage]]:
        """Input from the `Sample`, should be considered immutable."""
        return self._input

    @property
    def input_text(self) -> str:
        """
        Convenience function for accessing the initial input from the `Sample` as a string.

        If the `input` is a `List[ChatMessage]`, this will return the text from
        the last chat message
        """
        if isinstance(self._input, str):
            return self._input
        else:
            input = next(
                (message.text for message in reversed(self._input) if message.role == 'user'),
                None,
            )
            if input:
                return input
            else:
                raise ValueError('input_text requested from TaskState but none available')

    @property
    def user_prompt(self) -> ChatMessageUser:
        """User prompt for this state.

        Tasks are very general and can have may types of inputs.
        However, in many cases solvers assume they can interact with
        the state as a "chat" in a predictable fashion (e.g. prompt
        engineering solvers). This property enables easy read and
        write access to the user chat prompt. Raises an
        exception if there is no user prompt
        """
        prompt = next((m for m in reversed(self.messages) if m.role == 'user'), None)
        if prompt:
            return prompt
        else:
            raise ValueError('user_prompt requested from TaskState but none available')

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata from the `Sample` for this `TaskState`"""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict[str, Any]) -> None:
        self._metadata = metadata

    @property
    def messages(self) -> List[ChatMessage]:
        """
        Chat conversation history for sample.

        This will generally get appended to every time a `generate` call is made
        to the model. Useful for both debug and for solvers/scorers to assess
        model performance or choose the next step.
        """
        return self._messages

    @messages.setter
    def messages(self, messages: List[ChatMessage]) -> None:
        self._messages = messages

    @property
    def output(self) -> ModelOutput:
        """
        The 'final' model output once we've completed all solving.

        For simple evals this may just be the last `message` from the
        conversation history, but more complex solvers may set this directly.
        """
        return self._output

    @output.setter
    def output(self, output: ModelOutput) -> None:
        self._output = output

    @property
    def completed(self) -> bool:
        """Is the task completed."""
        return self._completed

    @completed.setter
    def completed(self, completed: bool) -> None:
        """Set the completed status."""
        self._completed = completed

    @property
    def target(self) -> str:
        """The scoring target for this `Sample`."""
        return self._target
