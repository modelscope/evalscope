"""MultiTurnArgs: Pydantic model for multi-turn conversation benchmark parameters.

Supports IntOrRange fields where a single integer is used directly and a
``[min, max]`` list triggers uniform random sampling via a seeded numpy RNG.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Optional, Union

# Single value or [min, max] range for uniform integer sampling
IntOrRange = Union[int, List[int]]


class MultiTurnArgs(BaseModel):
    """Parameters for multi-turn conversation datasets and benchmark runners.

    All ``IntOrRange`` fields accept either:
    * a single integer (used as-is), or
    * a two-element list ``[min, max]`` – sampled via
      ``np.random.randint(min, max + 1)`` when :meth:`sample_params` is called.

    Sampling relies on the global numpy RNG state seeded by
    ``seed_everything`` in ``main.py``.
    """

    # Conversation structure
    min_turns: int = 1
    """Minimum number of user turns per conversation."""

    max_turns: int = 5
    """Maximum number of user turns per conversation."""

    # Token-length controls (support range sampling)
    first_turn_length: IntOrRange = 65000
    """Target token count for the first user turn."""

    subsequent_turn_length: IntOrRange = 500
    """Target token increment per subsequent turn."""

    max_context_length: IntOrRange = 75000
    """Maximum context length (prompt tokens) allowed per conversation."""

    # Misc
    chars_per_token: float = 3.0
    """Estimated characters per token used for pre-filtering when no tokenizer is available."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator('first_turn_length', 'subsequent_turn_length', 'max_context_length', mode='before')
    @classmethod
    def _validate_int_or_range(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError(f'IntOrRange list must have exactly 2 elements [min, max], got {v}')
            if v[0] > v[1]:
                raise ValueError(f'IntOrRange min ({v[0]}) must be <= max ({v[1]})')
            return v
        raise TypeError(f'Expected int or [min, max] list, got {type(v).__name__}')

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(val: IntOrRange) -> int:
        """Sample a concrete integer from an IntOrRange value.

        Uses the global numpy RNG (seeded by ``seed_everything``).

        Args:
            val: Either a single int or a ``[min, max]`` list.

        Returns:
            Sampled integer value.
        """
        if isinstance(val, list):
            return int(np.random.randint(val[0], val[1] + 1))
        return int(val)

    def sample_params(self) -> dict:
        """Sample all IntOrRange fields and return a concrete parameter dict.

        Uses the global numpy RNG (seeded by ``seed_everything``).

        Returns:
            Dictionary with concrete integer values for all sampled fields.
        """
        return {
            'min_turns': self.min_turns,
            'max_turns': self.max_turns,
            'first_turn_length': self._sample(self.first_turn_length),
            'subsequent_turn_length': self._sample(self.subsequent_turn_length),
            'max_context_length': self._sample(self.max_context_length),
            'chars_per_token': self.chars_per_token,
        }
