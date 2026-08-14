import math
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()


class ScoreExtractor(ABC):
    """Strategy interface for extracting scores from judge responses."""

    @abstractmethod
    def extract(self, response: str) -> float:
        """Extract a numeric score from a text response."""


class PatternScoreExtractor(ScoreExtractor):
    """Extract categorical scores via regex pattern + mapping (e.g. A->1.0, B->0.0)."""

    def __init__(self, pattern: str, score_mapping: Optional[Dict[str, float]] = None):
        self.pattern = pattern
        self.score_mapping = score_mapping or {'A': 1.0, 'B': 0.0}

    def extract(self, response: str) -> float:
        """Use the score_pattern to extract categorical scores."""
        # strict standalone A/B matching using MULTILINE to handle simple outputs
        match = re.search(self.pattern, response, re.MULTILINE)
        if match:
            answer = match.group(1) if match.lastindex else match.group(0).strip()
            if answer not in self.score_mapping:
                logger.warning(
                    f"Matched '{answer}' for pattern '{self.pattern}' but no score mapping exists; "
                    f'returning 0.0. Response: {response}'
                )
                return 0.0
            return self.score_mapping[answer]
        else:
            logger.warning(f"No match found for pattern '{self.pattern}' in response: {response}")
            return 0.0


class NumericScoreExtractor(ScoreExtractor):
    """Extract numeric scores directly from response (e.g. [[0.75]])."""

    def __init__(self, pattern: str, clamp_min: float = 0.0, clamp_max: float = 1.0):
        self.pattern = pattern
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def extract(self, response: str) -> float:
        """Extract numeric score from the response using the score_pattern."""
        # Find all numeric tokens like [[0.5]] and take the last one (most decisive)
        matches = list(re.finditer(self.pattern, response))
        if not matches:
            logger.warning(f"No match found for pattern '{self.pattern}' in response: {response}")
            return 0.0

        def _clamped(val: float) -> float:
            """Clamp to [clamp_min, clamp_max], warning when the raw value was out of range."""
            # NaN compares false against every bound, so it would slip through the range
            # check below and propagate into aggregation, turning the whole metric into
            # NaN. Fail closed at clamp_min instead, as documented for [ERROR] responses.
            if math.isnan(val):
                logger.warning(f'Score NaN is not a usable rating, returning {self.clamp_min} in response: {response}')
                return self.clamp_min
            if val < self.clamp_min or val > self.clamp_max:
                clamped = max(self.clamp_min, min(self.clamp_max, val))
                logger.warning(
                    f'Score {val} out of range [{self.clamp_min}, {self.clamp_max}], '
                    f'clamped to {clamped} in response: {response}'
                )
                return clamped
            return val

        # iterate from last to first to pick the final rating
        for match in reversed(matches):
            # prefer captured groups
            for group in match.groups():
                if group is None:
                    continue
                try:
                    return _clamped(float(group))
                except (ValueError, TypeError):
                    continue
            # fallback: try entire match if groups fail
            try:
                return _clamped(float(match.group(0)))
            except (ValueError, TypeError):
                continue

        logger.warning(f'Failed to convert extracted values to float in response: {response}')
        return 0.0
