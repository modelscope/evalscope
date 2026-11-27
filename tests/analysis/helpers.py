"""Persisted review rows shared by facts and model-diagnosis tests."""

from typing import Optional

from evalscope.api.evaluator.cache import ReviewResult
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.metric import SampleScore, Score


def review(
    index: int,
    value: float = 0,
    output: str = 'ANSWER: B',
    prediction: str = 'B',
    group: Optional[str] = None,
    generation: Optional[int] = None,
) -> ReviewResult:
    return ReviewResult(
        index=index,
        target=['A'],
        messages=[ChatMessageUser(content='Select A.\u2028Keep this separator.'), ChatMessageAssistant(content=output)],
        sample_score=SampleScore(
            sample_id=index,
            group_id=group,
            generation_index=generation,
            score=Score(value={'accuracy': value}, prediction=output, extracted_prediction=prediction),
        ),
    )
