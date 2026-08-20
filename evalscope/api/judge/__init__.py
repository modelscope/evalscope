"""Native LLM judge subsystem: a JSON output contract and a single execution path.

Benchmark adapters declare judge cases and the JSON schema each reply must satisfy;
:class:`JudgeExecutor` owns every judge call, the retry boundaries, position swapping and
aggregation. Adapters never call a judge model and never parse a judge response.

The judge must be a model that can follow a structured-output instruction reliably; a weak judge
shows up as a high parse-failure rate in ``JudgeSummary.failures``.
"""
from .contracts import OutputContract, ParseResult
from .executor import JudgeExecutor, JudgeExecutorConfig
from .types import (
    CaseVerdict,
    JudgeAttempt,
    JudgeCase,
    JudgeContext,
    JudgeObservation,
    JudgeProtocol,
    JudgeRequest,
    JudgeReview,
    PairwiseOutcome,
    PairwisePlacementOutcome,
    Placement,
    ReducedVerdict,
)

__all__ = [
    'CaseVerdict',
    'JudgeAttempt',
    'JudgeCase',
    'JudgeContext',
    'JudgeExecutor',
    'JudgeExecutorConfig',
    'JudgeObservation',
    'JudgeProtocol',
    'JudgeRequest',
    'JudgeReview',
    'OutputContract',
    'ParseResult',
    'Placement',
    'PairwiseOutcome',
    'PairwisePlacementOutcome',
    'ReducedVerdict',
]
