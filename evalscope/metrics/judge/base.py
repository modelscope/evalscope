from abc import ABC, abstractmethod
from typing import List, Optional

from evalscope.api.messages import ChatMessage


class BaseJudge(ABC):
    """Abstract base class for all judge implementations."""

    @abstractmethod
    def judge(
        self,
        prompt: str = '',
        system_prompt: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None
    ) -> str:
        """Generate a judgment response."""

    @abstractmethod
    def build_prompt(self, pred: str, gold: str, question: Optional[str] = None) -> str:
        """Build the evaluation prompt."""

    @abstractmethod
    def get_score(self, response: str) -> float:
        """Extract a numeric score from the judge response."""
