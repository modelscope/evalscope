"""MultiTurnArgs: Pydantic model for multi-turn conversation benchmark parameters."""

from pydantic import BaseModel, ConfigDict


class MultiTurnArgs(BaseModel):
    """Parameters for multi-turn conversation datasets and benchmark runners.

    Note: ``min_turns`` and ``max_turns`` are top-level ``Arguments`` fields
    (``--min-turns`` / ``--max-turns``).  For ``swe_smith`` live construction
    the per-conversation turn count is sampled from ``[min_turns, max_turns]``
    using these token-length parameters to fill each turn.
    """

    # Token-length controls
    first_turn_length: int = 65000
    """Target token count for the first user turn."""

    subsequent_turn_length: int = 500
    """Target token increment per subsequent turn."""

    # Misc
    chars_per_token: float = 3.0
    """Estimated characters per token used for pre-filtering when no tokenizer is available."""

    num_workers: int = 4
    """Number of parallel worker processes for live conversation building (>1 uses multiprocessing.Pool)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def sample_params(self) -> dict:
        """Return a concrete parameter dict with all field values.

        Returns:
            Dictionary with field values for multi-turn construction.
        """
        return {
            'first_turn_length': self.first_turn_length,
            'subsequent_turn_length': self.subsequent_turn_length,
            'chars_per_token': self.chars_per_token,
            'num_workers': self.num_workers,
        }
