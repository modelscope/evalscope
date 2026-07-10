from __future__ import annotations

from typing import Dict, List, Type

from evalscope.perf.domain.errors import PerfConfigError
from evalscope.perf.workloads.base import WorkloadSource


class WorkloadRegistry:
    """Explicit registry for workload source classes."""

    def __init__(self) -> None:
        self._classes: Dict[str, Type[WorkloadSource]] = {}

    def register(self, source: Type[WorkloadSource]) -> Type[WorkloadSource]:
        name = source.meta.name
        if name in self._classes:
            raise PerfConfigError(f'Workload {name!r} is already registered')
        self._classes[name] = source
        return source

    def get(self, name: str) -> Type[WorkloadSource]:
        if name not in self._classes:
            raise PerfConfigError(f'Unknown workload {name!r}. Available workloads: {", ".join(self.names())}')
        return self._classes[name]

    def names(self) -> List[str]:
        return sorted(self._classes)


workload_registry = WorkloadRegistry()


def register_workload(source: Type[WorkloadSource]) -> Type[WorkloadSource]:
    return workload_registry.register(source)


def register_builtins() -> None:
    """Import built-in workload definitions exactly once."""
    if workload_registry.names():
        return
    from evalscope.perf.workloads.builtins import converters, prompt  # noqa: F401


register_builtins()
