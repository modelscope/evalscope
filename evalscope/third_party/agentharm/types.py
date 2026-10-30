"""Contracts the vendored AgentHarm tools and graders are written against.

These shapes stay beside the vendored sources so the port depends only on
``evalscope.api``: the pinned provenance constants, the suspended semantic
criterion and the sample identity the graders read as ``metadata``.
EvalScope-side configuration (dataset revisions, subsets, metric names) and
the judge output schemas live in :mod:`evalscope.benchmarks.agentharm.types`.
"""

from typing import Dict, Generator, List, Literal, Optional

from pydantic import BaseModel

SOURCE_SHA = 'd1adf47db09ade66824c1b8468b90be6fb5161c9'
UPSTREAM_VERSION = '8-B'


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
