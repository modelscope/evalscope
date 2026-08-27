# Copyright (c) Alibaba, Inc. and its affiliates.
"""Identity of a persisted evaluation report.

One evaluation run writes ``<root>/<run_id>/reports/<model_id>/<dataset>.json``, so a report is
addressed by exactly two names: the run directory and the model directory inside it. The datasets a
report covers are read from that directory and are therefore data, not identity.
"""

import os

from pydantic import BaseModel, field_validator

# Characters that would make a name span more than one path segment, or address its parent.
_INVALID_SEGMENTS = {'', '.', '..'}
_PATH_SEPARATORS = ('/', '\\', os.sep)

REF_SEPARATOR = '/'


class ReportRef(BaseModel):
    """Identity of one model's report inside one evaluation run."""

    run_id: str
    model_id: str

    @field_validator('run_id', 'model_id')
    @classmethod
    def _validate_segment(cls, value: str) -> str:
        """Keep every part of a reference a single, non-escaping path segment."""
        if value in _INVALID_SEGMENTS:
            raise ValueError(f'Report identifier must be a non-empty name, got {value!r}')
        if any(sep in value for sep in _PATH_SEPARATORS):
            raise ValueError(f'Report identifier must not contain a path separator, got {value!r}')
        return value

    @property
    def key(self) -> str:
        """Flat form used in URLs and as a client-side cache key."""
        return f'{self.run_id}{REF_SEPARATOR}{self.model_id}'

    @classmethod
    def parse(cls, value: str) -> 'ReportRef':
        """Parse the flat ``{run_id}/{model_id}`` form.

        Raises:
            ValueError: when the value is not two separator-joined names.
        """
        run_id, separator, model_id = value.partition(REF_SEPARATOR)
        if not separator:
            raise ValueError(f'Report reference must look like "run_id/model_id", got {value!r}')
        return cls(run_id=run_id, model_id=model_id)

    def __str__(self) -> str:
        return self.key
