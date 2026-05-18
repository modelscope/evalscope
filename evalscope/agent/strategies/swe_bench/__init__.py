"""SWE-bench specific agent strategies.

Importing this subpackage triggers ``@register_strategy`` decorators for:

- ``swe_bench_toolcall`` — mainline OpenAI function-calling protocol
  (mirrors mini-swe-agent's ``swebench.yaml``).
- ``swe_bench_backticks`` — textbased fenced ``mswea_bash_command``
  fallback (mirrors mini-swe-agent's ``swebench_backticks.yaml``).

Both strategies are dedicated to SWE-bench evaluation and use the literal
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` sentinel for submission.
"""

from .swe_bench_backticks import SweBenchBackticksStrategy
from .swe_bench_toolcall import SUBMIT_SENTINEL, SweBenchToolcallStrategy

__all__ = [
    'SweBenchToolcallStrategy',
    'SweBenchBackticksStrategy',
    'SUBMIT_SENTINEL',
]
