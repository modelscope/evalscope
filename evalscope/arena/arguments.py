import argparse
from dataclasses import dataclass

from evalscope.utils import BaseArgument


@dataclass
class Arguments(BaseArgument):
    """
    Arguments for the Arena Mode.
    This class defines the arguments required for running the arena battle mode,
    including output paths, comparison methods, and judge configurations.
    """
    output_paths: list[str]
    comparison_method: str = 'llm_judge'
    baseline_model: str = None
    output_dir: str = None
    judge_model_id: str = None
    judge_api_key: str = None
    judge_api_url: str = None
    judge_system_prompt: str = None
    judge_prompt_template: str = None
    metric_name: str = None

    def __post_init__(self):
        """
        Post-initialization to validate and set default values for arguments.
        """
        if not self.output_paths:
            raise ValueError('output_paths must be specified for arena mode')

        if self.comparison_method not in ['llm_judge', 'score']:
            raise ValueError("comparison_method must be 'llm_judge' or 'score'")

        # Set default judge model if not provided
        if self.comparison_method == 'llm_judge' and not self.judge_model_id:
            from evalscope.metrics.llm_judge import DEFAULT_JUDGE_MODEL
            self.judge_model_id = DEFAULT_JUDGE_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description='EvalScope Arena Mode for model comparison')
    add_argument(parser)
    return parser.parse_args()


def add_argument(parser: argparse.ArgumentParser):
    """
    Add arena-specific arguments to the parser.
    """
    parser.add_argument('--output_paths', nargs='+', required=True, help='Paths to evaluation output directories')
    parser.add_argument(
        '--comparison_method',
        type=str,
        default='llm_judge',
        choices=['llm_judge', 'score'],
        help='Method for comparing model outputs')
    parser.add_argument('--baseline_model', type=str, default=None, help='Name of baseline model for ELO calculation')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save battle results and report')
    parser.add_argument('--judge_model_id', type=str, default=None, help='Model ID for LLM judge')
    parser.add_argument('--judge_api_key', type=str, default=None, help='API key for LLM judge')
    parser.add_argument('--judge_api_url', type=str, default=None, help='API URL for LLM judge')
    parser.add_argument('--judge_system_prompt', type=str, default=None, help='System prompt for LLM judge')
    parser.add_argument('--judge_prompt_template', type=str, default=None, help='Prompt template for LLM judge')
    parser.add_argument('--metric_name', type=str, default=None, help='Name of metric to use for score comparison')
