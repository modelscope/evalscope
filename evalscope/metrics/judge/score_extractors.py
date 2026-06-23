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
            return self.score_mapping.get(answer, 0.0)
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

        # iterate from last to first to pick the final rating
        for match in reversed(matches):
            # prefer captured groups
            for group in match.groups():
                if group is None:
                    continue
                try:
                    val = float(group)
                    return max(self.clamp_min, min(self.clamp_max, val))
                except (ValueError, TypeError):
                    continue
            # fallback: try entire match if groups fail
            try:
                val = float(match.group(0))
                return max(self.clamp_min, min(self.clamp_max, val))
            except (ValueError, TypeError):
                continue

        logger.warning(f'Failed to convert extracted values to float in response: {response}')
        return 0.0
