from __future__ import annotations

import importlib
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional

from evalscope.perf.domain.observation import RequestObservation


class ObservationObserver:

    def observe(self, observation: RequestObservation) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Flush resources owned by the observer."""


class ProgressObserver(ObservationObserver):
    """Persist completed/failed/dropped/conversation counters atomically."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.completed = 0
        self.failed = 0
        self.dropped = 0
        self.conversations = 0

    def observe(self, observation: RequestObservation) -> None:
        if observation.is_warmup:
            return
        if observation.dropped:
            self.dropped += 1
        elif observation.success:
            self.completed += 1
        else:
            self.failed += 1
        if observation.is_last_turn:
            self.conversations += 1
        payload = {
            'completed': self.completed,
            'failed': self.failed,
            'dropped': self.dropped,
            'conversations_completed': self.conversations,
        }
        temporary = f'{self.path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as file:
            json.dump(payload, file)
        os.replace(temporary, self.path)


class VisualizerObserver(ObservationObserver):
    """Send visualizer updates on one background thread."""

    def __init__(self, provider: str, project: str, name: str) -> None:
        self.provider = provider
        self.project = project
        self.name = name
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'perf-{provider}')
        self._futures: List[Future] = []
        self._client = None
        self._futures.append(self._executor.submit(self._initialize))

    def _initialize(self) -> None:
        module = importlib.import_module(self.provider)
        if self.provider == 'wandb':
            self._client = module.init(project=self.project, name=self.name)
        elif self.provider == 'swanlab':
            self._client = module.init(project=self.project, experiment_name=self.name)
        else:
            self._client = module.Task.init(project_name=self.project, task_name=self.name)

    def observe(self, observation: RequestObservation) -> None:
        payload = {
            'requests/completed': int(observation.success),
            'requests/failed': int(not observation.success and not observation.dropped),
            'requests/dropped': int(observation.dropped),
            'latency': observation.latency,
            'ttft': observation.ttft,
        }
        self._futures.append(self._executor.submit(self._log, payload))

    def _log(self, payload) -> None:
        if self.provider in {'wandb', 'swanlab'}:
            importlib.import_module(self.provider).log(payload)
        elif self._client is not None:
            logger = self._client.get_logger()
            for key, value in payload.items():
                if value is not None:
                    logger.report_scalar('perf', key, value=float(value), iteration=0)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
