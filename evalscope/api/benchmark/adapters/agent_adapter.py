"""AgentAdapter: marker base class for agent-class benchmarks.

This class intentionally provides **no** behaviour beyond
:class:`DefaultDataAdapter`. It exists to act as a structural marker that
classifies a benchmark as belonging to the *agent* family — used by
documentation generators (see ``generate_dataset_md.py`` which calls
``issubclass(adapter_cls, AgentAdapter)``) and downstream tooling.

Two extension modes are supported:

1. **Custom multi-turn loop** (e.g. ``tau_bench``, ``bfcl_v3/v4``,
   ``general_fc``): subclass :class:`AgentAdapter` directly and override
   :meth:`_on_inference` to drive the benchmark's bespoke loop. The
   built-in :class:`evalscope.api.agent.AgentLoop` is **not** used.

2. **Generic AgentLoop driving** (e.g. ``swe_bench_*_agentic``): subclass
   :class:`AgentLoopAdapter` instead, which derives from this class and
   wires together :class:`AgentLoop` + ``build_*`` hooks.
"""

from .default_data_adapter import DefaultDataAdapter


class AgentAdapter(DefaultDataAdapter):
    """Marker base class for agent-class benchmarks.

    Subclasses are free to override :meth:`_on_inference` to plug in their
    own multi-turn driver. For benchmarks that want the standard
    :class:`AgentLoop` orchestration, subclass :class:`AgentLoopAdapter`
    instead.
    """

    pass
