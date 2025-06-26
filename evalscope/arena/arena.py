"""
Arena battle CLI for model comparison
"""
import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

from evalscope.arena.arena_utils import run_arena_battle
from evalscope.arena.config_arena import ArenaConfig
from evalscope.utils.logger import get_logger

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description='Arena Battle Mode')

    # Basic configuration
    parser.add_argument('--output_paths', nargs='+', required=True, help='Paths to evaluation output directories')
    parser.add_argument(
        '--comparison_method',
        type=str,
        default='llm_judge',
        choices=['llm_judge', 'score'],
        help='Method for comparing model outputs')
    parser.add_argument('--baseline_model', type=str, default=None, help='Name of baseline model for ELO calculation')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save battle results and report')

    # LLM Judge configuration
    parser.add_argument('--judge_model_id', type=str, default=None, help='Model ID for LLM judge')
    parser.add_argument('--judge_api_key', type=str, default=None, help='API key for LLM judge')
    parser.add_argument('--judge_api_url', type=str, default=None, help='API URL for LLM judge')
    parser.add_argument('--judge_system_prompt', type=str, default=None, help='System prompt for LLM judge')
    parser.add_argument('--judge_prompt_template', type=str, default=None, help='Prompt template for LLM judge')

    # Score comparison configuration
    parser.add_argument('--metric_name', type=str, default=None, help='Name of metric to use for score comparison')

    # Logging configuration
    parser.add_argument(
        '--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)

    # Configure arena
    config = ArenaConfig()
    config.output_paths = args.output_paths
    config.comparison_method = args.comparison_method
    config.baseline_model = args.baseline_model
    config.judge_model_id = args.judge_model_id
    config.judge_api_key = args.judge_api_key
    config.judge_api_url = args.judge_api_url
    config.judge_system_prompt = args.judge_system_prompt
    config.judge_prompt_template = args.judge_prompt_template
    config.metric_name = args.metric_name

    # Set up judge config if using LLM judge
    judge_config = None
    if config.comparison_method == 'llm_judge':
        judge_config = {
            'model_id': config.judge_model_id,
            'api_key': config.judge_api_key,
            'api_url': config.judge_api_url,
            'system_prompt': config.judge_system_prompt,
            'prompt_template': config.judge_prompt_template
        }
        # Remove None values
        judge_config = {k: v for k, v in judge_config.items() if v is not None}

    # Run arena battle
    results = run_arena_battle(
        output_paths=config.output_paths,
        comparison_method=config.comparison_method,
        baseline_model=config.baseline_model,
        judge_config=judge_config,
        metric_name=config.metric_name,
        output_dir=args.output_dir)

    # Print summary
    print('\n=== Arena Battle Summary ===')
    print(f"Total battles: {len(results['battles'])}")
    print(f"Models compared: {', '.join(results['elo_ratings'].keys())}")
    print(f"Report saved to: {results['report_path']}")

    # Print leaderboard
    print('\n=== Overall Leaderboard ===')
    print(results['leaderboards']['overall'].to_string(index=False))

    return 0


if __name__ == '__main__':
    sys.exit(main())
