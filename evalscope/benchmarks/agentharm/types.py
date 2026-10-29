"""Small contracts for fixed AgentHarm tasks and their native judge cases."""

from typing import Dict, Generator, List, Literal, Optional

from pydantic import BaseModel, Field

SOURCE_SHA = 'd1adf47db09ade66824c1b8468b90be6fb5161c9'
DATASET_REVISION = 'e23b3fe60a0da9037314b88e5ee3a0c054970dad'
MS_DATASET_REVISION = '9252a5682b6a2cf034ca103eeee5cdc2f6db1cd1'
UPSTREAM_VERSION = '8-B'
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


class AgentHarmSampleMeta(BaseModel):
    """Serializable task identity and the exact tool selection shown to the agent."""

    task_kind: Literal['harmful', 'benign']
    behavior_id: str
    id_original: str
    category: str
    split: Literal['test_public', 'val']
    grading_function: str
    target_functions: List[str]
    tool_order: List[str]
    detailed_prompt: bool
    hint_included: bool
    dataset_checksum: str
    dataset_revision: Optional[str] = None
    source_sha: str = SOURCE_SHA
    upstream_version: str = UPSTREAM_VERSION
    evaluation_version: str = 'v1.0'
    tool_order_seed: int


class SemanticCheck(BaseModel):
    """A task's semantic criterion, before any judge request is made."""

    prompt: str
    text: str


Rubric = Generator[SemanticCheck, str, Dict[str, float]]


class SemanticVerdict(BaseModel):
    """Official pass/fail labels with the native JSON output contract."""

    verdict: Literal['pass', 'fail']
    reasoning: str = ''


class RefusalVerdict(BaseModel):
    """Whether a complete assistant message constitutes a refusal."""

    refusal: bool
    reasoning: str = ''
