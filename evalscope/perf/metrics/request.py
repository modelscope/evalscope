from dataclasses import dataclass

from evalscope.perf.domain.observation import RequestObservation


@dataclass
class RequestMetricsAggregator:
    """Lightweight online counters; exact summaries are produced from ResultStore."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    dropped: int = 0

    def feed(self, observation: RequestObservation) -> None:
        if observation.is_warmup:
            return
        self.total += 1
        if observation.dropped:
            self.dropped += 1
        elif observation.success:
            self.succeeded += 1
        else:
            self.failed += 1
