"""Shared doubles and sample builders for judge adapter tests."""

from typing import List

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage, ContentText
from evalscope.api.model import ModelOutput
from evalscope.constants import JudgeScoreType
from evalscope.metrics.judge.llm_judge import DEFAULT_PROMPT_TEMPLATE, LLMJudge


class ScriptedJudge:
    """Return scripted replies while recording the actual judge requests."""

    score_type = JudgeScoreType.PATTERN
    score_mapping = {'A': 1.0, 'B': 0.0}
    prompt_template = DEFAULT_PROMPT_TEMPLATE
    system_prompt = None
    build_prompt = LLMJudge.build_prompt

    def __init__(self, replies: List[str]) -> None:
        self.replies = replies
        self.judge_id = self.model_id = 'scripted'
        self.calls: List[List[ChatMessage]] = []

    def generate(self, messages: List[ChatMessage]) -> ModelOutput:
        """Return the next reply, repeating the final reply when the script is exhausted."""
        self.calls.append(messages)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return ModelOutput.from_content('scripted', [ContentText(text=reply)])


def make_state(prediction: str, target: str) -> TaskState:
    """Build a completed QA state for a controlled prediction and reference."""
    sample = Sample(id=0, input='Who wrote Hamlet?', target=target, metadata={})
    return TaskState(model='m', sample=sample, output=ModelOutput.from_content('m', prediction), completed=True)
