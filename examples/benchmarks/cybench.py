"""Run native Cybench smoke tasks against a real model and inspect their persisted artifacts."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

SMOKE_TASKS = [
    'benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic',
    'benchmark/hackthebox/cyber-apocalypse-2024/web/[Very Easy] Flag Command',
    'benchmark/LosFuzzys/GlacierCTF2023_writeups/crypto/SLCG',
    'benchmark/LosFuzzys/GlacierCTF2023_writeups/intro/skilift',
]


def main() -> None:
    """Evaluate selected tasks with pinned official runtimes and write a compact acceptance receipt."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-ids', nargs='+', default=SMOKE_TASKS)
    parser.add_argument('--image', help='Override sandbox.default_config.image; defaults to the official Kali image.')
    parser.add_argument('--output-dir', default='outputs/cybench-official-smoke')
    parser.add_argument('--use-cache', help='Resume completed predictions from a previous run directory.')
    parser.add_argument('--max-steps', type=int, default=15)
    args = parser.parse_args()
    key = os.environ['DASHSCOPE_API_KEY']
    output = Path(args.output_dir).resolve()
    config = TaskConfig(
        model='qwen3.8-flash', eval_type='openai_api',
        api_url=os.environ.get('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
        api_key=key, datasets=['cybench'],
        dataset_args={'cybench': {'extra_params': {
            'task_ids': args.task_ids,
        }}},
        sandbox={'default_config': {'image': args.image}} if args.image else None,
        agent_config=NativeAgentConfig(max_steps=args.max_steps),
        generation_config={
            'temperature': 0, 'max_tokens': 2048, 'timeout': 120,
            'extra_body': {'enable_thinking': False},
        },
        eval_batch_size=1, work_dir=str(output), no_timestamp=True, use_cache=args.use_cache,
    )
    try:
        report = run_task(config)['cybench']
        records = []
        for path in sorted((output / 'predictions/qwen3.8-flash').glob('*.jsonl')):
            for line in path.read_text().splitlines():
                prediction = json.loads(line)
                record = prediction['metadata']['cybench_run']
                events = prediction['agent_trace']['events']
                assert events and prediction['messages']
                assert all(e['payload']['model_input'] for e in events if e['type'] == 'model_generate')
                records.append({
                    'task': record['task']['path'], 'termination': record['termination'],
                    'runtime': record['runtime'], 'commands': len(record['executions']),
                    'events': len(events), 'submission': record['submission'],
                })
        assert len(records) == len(args.task_ids)
        assert all(metric.num == len(records) for metric in report.metrics)
        receipt = {'model': config.model, 'tasks': records, 'metrics': {
            metric.identity.name: {'score': metric.score, 'num': metric.num} for metric in report.metrics
        }}
        (output / 'runtime_acceptance.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('CYBENCH_REAL_ACCEPTANCE', json.dumps({
            'tasks': [(r['task'], r['termination']) for r in records], 'metrics': receipt['metrics'],
        }))
    finally:
        # Keep provider credentials out of downloadable validation artifacts, including failure logs.
        for path in output.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.yaml', '.yml', '.log'):
                content = path.read_text()
                if key in content:
                    path.write_text(content.replace(key, '[REDACTED]'))


if __name__ == '__main__':
    main()
