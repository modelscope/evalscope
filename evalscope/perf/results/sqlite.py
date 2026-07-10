from __future__ import annotations

import json
import os
import sqlite3
from typing import Iterator, Optional

from evalscope.perf.domain.errors import ResultStoreError
from evalscope.perf.domain.observation import RequestObservation
from evalscope.perf.results.store import ResultStore

SCHEMA_VERSION = 1


class SQLiteResultStore(ResultStore):
    """Versioned JSON-safe observation store."""

    def __init__(self, path: str, commit_interval: int = 1000) -> None:
        self.path = path
        self.commit_interval = commit_interval
        self._connection: Optional[sqlite3.Connection] = None
        self._pending = 0

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            self._connection = sqlite3.connect(self.path)
            self._connection.execute('CREATE TABLE schema_info(version INTEGER NOT NULL)')
            self._connection.execute('INSERT INTO schema_info(version) VALUES (?)', (SCHEMA_VERSION, ))
            self._connection.execute(
                '''CREATE TABLE observations(
                    request_id TEXT PRIMARY KEY,
                    is_warmup INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    dropped INTEGER NOT NULL,
                    trace_id TEXT,
                    start_time REAL,
                    completed_time REAL,
                    payload TEXT NOT NULL
                )'''
            )
            self._connection.commit()
        except sqlite3.Error as e:
            raise ResultStoreError(f'Unable to create result store {self.path}: {e}') from e

    def write(self, observation: RequestObservation) -> None:
        if self._connection is None:
            raise ResultStoreError('Result store is not open')
        try:
            self._connection.execute(
                'INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    observation.request_id,
                    int(observation.is_warmup),
                    int(observation.success),
                    int(observation.dropped),
                    observation.trace_id,
                    observation.start_time,
                    observation.completed_time,
                    observation.model_dump_json(),
                ),
            )
            self._pending += 1
            if self._pending >= self.commit_interval:
                self._connection.commit()
                self._pending = 0
        except sqlite3.Error as e:
            raise ResultStoreError(f'Unable to write observation: {e}') from e

    def observations(self, include_warmup: bool = False) -> Iterator[RequestObservation]:
        if self._connection is None:
            raise ResultStoreError('Result store is not open')
        query = 'SELECT payload FROM observations'
        if not include_warmup:
            query += ' WHERE is_warmup = 0'
        query += ' ORDER BY rowid'
        for (payload, ) in self._connection.execute(query):
            yield RequestObservation.model_validate_json(payload)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.commit()
            self._connection.close()
            self._connection = None
