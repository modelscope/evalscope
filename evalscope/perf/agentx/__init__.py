# SPDX-License-Identifier: Apache-2.0
"""AgentX (InferenceX) scenario for EvalScope Perf.

This module wraps the official AIPerf runner for the
``inferencex-agentx-mvp`` scenario instead of reimplementing the AgentX
session-tree scheduler inside EvalScope. See issue #1706 for the design
rationale.

Key properties:

- AIPerf stays an **optional** dependency: it is resolved from the user
  environment (or an explicit executable path) and is never added to the
  default Perf runtime dependency set.
- The exact AIPerf version/commit is recorded with every result for
  provenance.
- Raw AIPerf artifacts are preserved unchanged next to the normalized
  result.
- The normalized result distinguishes canonical (comparable) runs from
  smoke runs, and surfaces ``submission_valid`` plus its reasons.
"""

from .config import AgentxArguments, validate_agentx_arguments
from .runner import AIPerfRunner, build_aiperf_command, parse_profile_export
from .result import AgentxResult, load_agentx_result

__all__ = [
    'AgentxArguments',
    'AgentxResult',
    'AIPerfRunner',
    'build_aiperf_command',
    'load_agentx_result',
    'parse_profile_export',
    'validate_agentx_arguments',
]
