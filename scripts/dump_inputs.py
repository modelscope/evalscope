#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Iterable, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evalscope.api.messages import ChatMessage, messages_pretty_str  # noqa: E402
from evalscope.api.registry import get_benchmark  # noqa: E402
from evalscope.config import TaskConfig  # noqa: E402
from evalscope.constants import EvalType  # noqa: E402
from evalscope.run import run_task  # noqa: E402
from evalscope.models.utils.openai import openai_prompt_from_messages  # noqa: E402


def build_task_config(args: argparse.Namespace) -> TaskConfig:
    dataset_args = {args.benchmark: {}}
    if args.subset:
        dataset_args[args.benchmark]['subset_list'] = [args.subset]
    if args.few_shot_num is not None:
        dataset_args[args.benchmark]['few_shot_num'] = args.few_shot_num
    if args.train_split:
        dataset_args[args.benchmark]['train_split'] = args.train_split
    if args.eval_split:
        dataset_args[args.benchmark]['eval_split'] = args.eval_split

    return TaskConfig(
        model=args.model,
        api_url=args.api_url,
        datasets=[args.benchmark],
        dataset_args=dataset_args,
        dataset_hub=args.dataset_hub,
        limit=args.limit,
    )


def iter_prepared_messages(benchmark, max_samples: int) -> Iterable[Tuple[str, List[ChatMessage]]]:
    benchmark.test_dataset, benchmark.fewshot_dataset = benchmark.load()
    emitted = 0
    for subset, dataset in benchmark.test_dataset.items():
        for sample in dataset:
            if isinstance(sample.input, str):
                messages = benchmark.process_sample_str_input(sample, subset)
            else:
                messages = benchmark.process_sample_messages_input(sample, subset)
            yield subset, messages
            emitted += 1
            if max_samples and emitted >= max_samples:
                return


def message_to_dict(message: ChatMessage) -> dict:
    if hasattr(message, 'model_dump'):
        return message.model_dump(exclude_none=True)
    return dict(message)  # type: ignore[arg-type]


def _load_env_value(key: str) -> Optional[str]:
    if not key:
        return None
    if key in os.environ:
        return os.environ.get(key)
    try:
        from dotenv import dotenv_values  # type: ignore
    except Exception:
        return None
    env = dotenv_values(os.path.join(ROOT, '.env'))
    return env.get(key)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Print EvalScope-formatted inputs (few-shot handled by EvalScope).'
    )
    parser.add_argument('--benchmark', default='mmlu', help='Benchmark name (default: mmlu)')
    parser.add_argument('--subset', default=None, help='Subset name (optional)')
    parser.add_argument('--few-shot-num', type=int, default=None, help='Override few-shot count')
    parser.add_argument('--train-split', default=None, help='Override train split for few-shot')
    parser.add_argument('--eval-split', default=None, help='Override eval split')
    parser.add_argument('--dataset-hub', default='modelscope', help='Dataset hub (modelscope or huggingface)')
    parser.add_argument('--limit', type=float, default=1, help='Limit samples per subset (int or fraction)')
    parser.add_argument('--max-samples', type=int, default=1, help='Max samples to print')
    parser.add_argument('--format', choices=['pretty', 'json'], default='pretty', help='Output format')
    parser.add_argument('--show-prompt', action='store_true', help='Include completion prompt string')
    parser.add_argument('--run', action='store_true', help='Run evaluation after printing inputs')
    parser.add_argument('--model', default='openai/gpt-4.1', help='Model name for service eval')
    parser.add_argument('--api-url', default='https://openrouter.ai/api/v1/completions',
                        help='OpenAI-compatible base URL')
    parser.add_argument('--api-key', default="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", help='API key for service eval (optional)')
    parser.add_argument('--api-key-env', default='DASHSCOPE_API_KEY',
                        help='Env var name to load API key from (default: DASHSCOPE_API_KEY)')
    parser.add_argument('--completion-endpoint', action='store_true',
                        help='Force /completions endpoint for OpenAI-compatible API')
    args = parser.parse_args()

    task_cfg = build_task_config(args)
    benchmark = get_benchmark(args.benchmark, task_cfg)

    for subset, messages in iter_prepared_messages(benchmark, args.max_samples):
        if args.format == 'json':
            payload = {
                'subset': subset,
                'messages': [message_to_dict(m) for m in messages],
            }
            if args.show_prompt:
                payload['prompt'] = openai_prompt_from_messages(messages)
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f'=== {subset} ===')
            print(messages_pretty_str(messages))
            if args.show_prompt:
                print('\n---\nPrompt:')
                print(openai_prompt_from_messages(messages))
            print('')

    if args.run:
        api_key = args.api_key or _load_env_value(args.api_key_env)
        if not api_key:
            raise SystemExit(f'API key not found. Set {args.api_key_env} or pass --api-key.')

        model_args = {}
        run_cfg = TaskConfig(
            model=args.model,
            api_url=args.api_url,
            api_key=api_key,
            eval_type=EvalType.SERVICE,
            datasets=task_cfg.datasets,
            dataset_args=task_cfg.dataset_args,
            dataset_hub=task_cfg.dataset_hub,
            limit=task_cfg.limit,
            eval_batch_size=1,
            stream=True,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': 4096,
            },
            model_args=model_args,
        )
        run_task(task_cfg=run_cfg)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
