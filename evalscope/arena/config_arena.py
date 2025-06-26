"""
Arena configuration for model comparison
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from evalscope.config import TaskConfig


@dataclass
class ArenaConfig(TaskConfig):
    """
    Configuration for arena battle mode.
    """
    # Arena-specific arguments
    output_paths: List[str] = field(default_factory=list)
    comparison_method: str = 'llm_judge'  # "llm_judge" or "score"
    baseline_model: Optional[str] = None
    judge_model_id: Optional[str] = None
    judge_api_key: Optional[str] = None
    judge_api_url: Optional[str] = None
    judge_system_prompt: Optional[str] = None
    judge_prompt_template: Optional[str] = None
    metric_name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

        # Set default values
        if not self.output_paths:
            raise ValueError('output_paths must be specified for arena mode')

        if self.comparison_method not in ['llm_judge', 'score']:
            raise ValueError("comparison_method must be 'llm_judge' or 'score'")

        # Configure judge if using llm_judge method
        if self.comparison_method == 'llm_judge' and not self.judge_model_id:
            from evalscope.metrics.llm_judge import DEFAULT_JUDGE_MODEL
            self.judge_model_id = DEFAULT_JUDGE_MODEL
