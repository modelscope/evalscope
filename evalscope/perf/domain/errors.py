class PerfError(Exception):
    """Base exception for all performance benchmark failures."""


class PerfConfigError(PerfError):
    """Raised when a performance benchmark configuration is invalid."""


class PerfUsageError(PerfError):
    """Raised when a public API is used from an unsupported execution context."""


class PerfRunError(PerfError):
    """Raised when a benchmark run cannot complete."""

    def __init__(self, run_id: str, stage: str, message: str) -> None:
        self.run_id = run_id
        self.stage = stage
        super().__init__(f'Run {run_id} failed during {stage}: {message}')


class TransportError(PerfError):
    """Raised for transport infrastructure failures."""


class ResultStoreError(PerfError):
    """Raised when benchmark observations cannot be persisted."""


class ResultAlreadyExistsError(ResultStoreError):
    """Raised when a run output already exists and overwrite is disabled."""
