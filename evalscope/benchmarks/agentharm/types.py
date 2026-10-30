"""EvalScope-side AgentHarm configuration and the native judge output schemas.

The contracts the vendored sources are written against (sample identity, the
suspended semantic criterion and the pinned provenance constants) live in
:mod:`evalscope.third_party.agentharm.types`.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

DATASET_REVISION = 'e23b3fe60a0da9037314b88e5ee3a0c054970dad'
MS_DATASET_REVISION = '9252a5682b6a2cf034ca103eeee5cdc2f6db1cd1'
CATEGORIES = ('Disinformation', 'Harassment', 'Drugs', 'Fraud', 'Hate', 'Cybercrime', 'Sexual', 'Copyright')
METRICS = ('task_completion_score', 'full_completion_rate', 'refusal_rate', 'non_refusal_completion_score')


class Behavior(BaseModel):
    """One row of the official behaviors array."""

    id: str = Field(min_length=1)
    id_original: str = Field(min_length=1)
    name: str
    category: str
    prompt: str = Field(min_length=1)
    target_functions: List[str] = Field(min_length=1)
    grading_function: str
    detailed_prompt: bool
    hint_included: bool


class SemanticVerdict(BaseModel):
    """Official pass/fail labels with the native JSON output contract."""

    verdict: Literal['pass', 'fail']
    reasoning: str = ''


class RefusalVerdict(BaseModel):
    """Whether a complete assistant message constitutes a refusal."""

    refusal: bool
    reasoning: str = ''
