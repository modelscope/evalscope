import json
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator
from typing import Any, Dict, Iterator, List, Optional

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset
from evalscope.perf.utils.body_meta import (
    BODY_META_ARRIVAL_OFFSET,
    BODY_META_HEADERS,
    BODY_META_IS_WARMUP,
    BODY_META_REQUEST_ID,
)
from evalscope.utils.logger import get_logger

logger = get_logger()

RESERVED_BODY_KEYS = frozenset({BODY_META_HEADERS, BODY_META_REQUEST_ID, BODY_META_ARRIVAL_OFFSET, BODY_META_IS_WARMUP})


class WorkloadTraceArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    speed: float = 1.0
    model_override: Optional[str] = None
    model_mapping: Optional[Dict[str, str]] = None

    @field_validator('speed')
    @classmethod
    def _speed_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f'speed must be > 0, got {v}')
        return v


def _normalise_timestamp(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        from datetime import datetime, timezone
        s = v.strip()
        if s.endswith('Z') or s.endswith('z'):
            s = s[:-1] + '+00:00'
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception as e:
            raise ValueError(f'cannot parse timestamp: {v!r}') from e
    raise TypeError(f'timestamp must be a number or ISO-8601 string, got {type(v).__name__}')


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra='forbid')

    body: Dict[str, Any]
    timestamp: float
    headers: Optional[Dict[str, str]] = None
    request_id: Optional[str] = None

    @field_validator('timestamp', mode='before')
    @classmethod
    def _parse_timestamp(cls, v: Any) -> float:
        return _normalise_timestamp(v)

    @field_validator('body', mode='before')
    @classmethod
    def _parse_body(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f'body is a string but not valid JSON: {e}')
            if not isinstance(parsed, dict):
                raise ValueError(f'body JSON string must decode to a dict, got {type(parsed).__name__}')
            return parsed
        return v


@register_dataset('workload_trace')
class WorkloadTraceDatasetPlugin(DatasetPluginBase):
    """Replay recorded production traffic traces with original arrival pattern.

    Each line in the JSONL dataset file must be a JSON object with:

    - ``body`` (dict or JSON string, required): Complete request body.
    - ``timestamp`` (number or ISO-8601 string, required): Arrival time.
    - ``headers`` (dict, optional): Per-request HTTP headers.
    - ``request_id`` (str, optional): ID for result correlation.

    Requires open-loop mode (``--open-loop``).
    """

    provides_arrival_schedule = True
    requires_model = False

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

        if not query_parameters.open_loop:
            raise ValueError('workload_trace requires open-loop mode. '
                             'Add --open-loop to your command.')

        if not query_parameters.dataset_path:
            raise ValueError('workload_trace requires --dataset-path pointing to a JSONL trace file.')

        raw_args = query_parameters.dataset_args or {}
        self._config = WorkloadTraceArgs(**raw_args)
        self._records: List[TraceRecord] = []
        self._load_and_validate()
        self._schedule = self._compute_schedule()

        n = len(self._records)

        # --number: if user didn't pass it, use all records; if passed, truncate.
        if 'number' in query_parameters.model_fields_set:
            n = min(query_parameters._current_number(), n)
            self._records = self._records[:n]
            self._schedule = self._schedule[:n]
        query_parameters.number = n

        # --model: if user explicitly passed it, treat as model_override fallback.
        if 'model' in query_parameters.model_fields_set:
            if not self._config.model_override and not self._config.model_mapping:
                self._config.model_override = query_parameters.model

        self._warmup_count = min(query_parameters.warmup_count, n)

        query_parameters.warmup_num = 0

        warmup_msg = f', warmup={self._warmup_count}' if self._warmup_count else ''
        logger.info(f'workload_trace: loaded {n} records, '
                    f'speed={self._config.speed}x{warmup_msg}')

    def _load_and_validate(self) -> None:
        errors: List[str] = []
        path = self.query_parameters.dataset_path

        with open(path, 'r', encoding='utf-8') as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f'line {line_num}: invalid JSON: {e}')
                    continue

                if not isinstance(data, dict):
                    errors.append(f'line {line_num}: expected JSON object, got {type(data).__name__}')
                    continue

                if 'body' not in data:
                    errors.append(f'line {line_num}: missing required field "body"')
                    continue
                if not isinstance(data['body'], (dict, str)):
                    errors.append(f'line {line_num}: "body" must be a dict or JSON string')
                    continue
                if 'timestamp' not in data:
                    errors.append(f'line {line_num}: missing required field "timestamp"')
                    continue
                if not isinstance(data['timestamp'], (int, float, str)):
                    errors.append(f'line {line_num}: "timestamp" must be a number or ISO-8601 string')
                    continue

                body_for_check = data['body']
                if isinstance(body_for_check, str):
                    try:
                        body_for_check = json.loads(body_for_check)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if isinstance(body_for_check, dict):
                    collisions = RESERVED_BODY_KEYS & body_for_check.keys()
                    if collisions:
                        errors.append(
                            f'line {line_num}: body contains reserved key(s) {collisions}; '
                            f'remove them from the trace body'
                        )
                        continue

                try:
                    self._records.append(TraceRecord(**data))
                except Exception as e:
                    errors.append(f'line {line_num}: {e}')

        if errors:
            raise ValueError(f'workload_trace validation failed ({len(errors)} error(s)):\n' + '\n'.join(errors))

        if not self._records:
            raise ValueError('workload_trace: dataset is empty')

        for i in range(1, len(self._records)):
            if self._records[i].timestamp < self._records[i - 1].timestamp:
                raise ValueError(
                    f'Timestamps must be monotonically non-decreasing: '
                    f'record {i + 1} (t={self._records[i].timestamp}) < '
                    f'record {i} (t={self._records[i - 1].timestamp})'
                )

    def _compute_schedule(self) -> np.ndarray:
        if not self._records:
            return np.array([])
        t0 = self._records[0].timestamp
        return np.array([(r.timestamp - t0) / self._config.speed for r in self._records])

    def _rewrite_model(self, body: Dict[str, Any]) -> None:
        original = body.get('model')
        mapping = self._config.model_mapping
        override = self._config.model_override

        if mapping and original in mapping:
            body['model'] = mapping[original]
        elif override is not None:
            body['model'] = override

    def build_messages(self) -> Iterator[Dict[str, Any]]:
        for i, record in enumerate(self._records):
            body = dict(record.body)
            self._rewrite_model(body)

            body[BODY_META_ARRIVAL_OFFSET] = float(self._schedule[i])
            if i < self._warmup_count:
                body[BODY_META_IS_WARMUP] = True
            if record.headers:
                body[BODY_META_HEADERS] = dict(record.headers)
            if record.request_id:
                body[BODY_META_REQUEST_ID] = record.request_id

            yield body
