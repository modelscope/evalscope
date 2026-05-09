"""MultiTurnArgs: Pydantic model for multi-turn conversation benchmark parameters."""

import random
from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Union

# Type alias: either an int or a [min, max] range for uniform sampling.
IntOrRange = Union[int, List[int]]


def _sample_int_or_range(v: IntOrRange) -> int:
    """Return a uniformly sampled int from a [min, max] range, or the value itself."""
    if isinstance(v, list):
        return random.randint(v[0], v[1])
    return v


def _get_range_upper(v: IntOrRange) -> int:
    """Return the upper bound of an IntOrRange (upper element of list, or the int itself)."""
    return v[1] if isinstance(v, list) else v


class MultiTurnArgs(BaseModel):
    """Parameters for multi-turn conversation datasets and benchmark runners.

    Note: ``min_turns`` and ``max_turns`` are top-level ``Arguments`` fields
    (``--min-turns`` / ``--max-turns``).  For ``swe_smith`` live construction
    the per-conversation turn count is sampled from ``[min_turns, max_turns]``
    using these token-length parameters to fill each turn.
    """

    # Token-length controls
    first_turn_length: IntOrRange = 65000
    """Target token count for the first user turn.

    Accepts an int or a ``[min, max]`` list for uniform sampling on each call to
    :meth:`sample_params`.
    """

    subsequent_turn_length: IntOrRange = 500
    """Target token increment per subsequent turn.

    Accepts an int or a ``[min, max]`` list for uniform sampling on each call to
    :meth:`sample_params`.
    """

    # Misc
    chars_per_token: float = 3.0
    """Estimated characters per token used for pre-filtering when no tokenizer is available."""

    num_workers: int = 4
    """Number of parallel worker processes for live conversation building (>1 uses multiprocessing.Pool)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('first_turn_length', 'subsequent_turn_length', mode='before')
    @classmethod
    def _validate_int_or_range(cls, v):
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError(f'IntOrRange list must have exactly 2 elements [min, max], got {v}')
            if v[0] > v[1]:
                raise ValueError(f'IntOrRange list min must be <= max, got {v}')
            if v[0] < 0:
                raise ValueError(f'IntOrRange list values must be >= 0, got {v}')
        return v

    def sample_params(self) -> dict:
        """Return a concrete parameter dict, sampling IntOrRange fields.

        Each call independently samples ``first_turn_length`` and
        ``subsequent_turn_length`` when they are ranges.

        Returns:
            Dictionary with concrete int values for multi-turn construction.
        """
        return {
            'first_turn_length': _sample_int_or_range(self.first_turn_length),
            'subsequent_turn_length': _sample_int_or_range(self.subsequent_turn_length),
            'chars_per_token': self.chars_per_token,
            'num_workers': self.num_workers,
        }
