"""SWE-bench benchmark adapters.

Importing this package triggers ``@register_benchmark`` decorators for both
the original oracle (single-turn) and agentic (multi-turn) variants.
"""

from . import swe_bench_adapter  # noqa: F401 - register oracle adapters
from . import swe_bench_agentic_adapter  # noqa: F401 - register agentic adapters
