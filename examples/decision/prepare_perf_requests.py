"""Export representative benchmark inputs as complete, gold-free decision perf requests."""

import argparse
import hashlib
import json
from pathlib import Path

from evalscope import TaskConfig
from evalscope.api.model.decision import DecisionInput
from evalscope.api.registry import get_benchmark


def main() -> None:
    """Keep request sets and their lengths identical across concurrency levels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', choices=['systemone', 'autotrust'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for kind, dataset in [('choice', 'banking77'), ('noul', 'boolq'), ('score', 'stsb')]:
        extra = {'extra_params': {'decision_kind': 'noul'}} if kind == 'noul' else {}
        config = TaskConfig(model=args.model, eval_type=args.api + '_api', datasets=[dataset],
                            dataset_args={dataset: extra}, limit=args.samples)
        adapter = get_benchmark(dataset, config)
        bodies = []
        for samples in adapter.load_dataset().values():
            for sample in samples:
                user = next(message for message in sample.input if message.role == 'user')
                question = DecisionInput.model_validate(user.internal['systemone'])
                body = question.systemone_payload(args.model) if args.api == 'systemone' else question.autotrust_payload()
                bodies.append(json.dumps(body, ensure_ascii=False))
        path = args.output / f'{kind}.jsonl'
        raw = ('\n'.join(bodies) + '\n').encode()
        path.write_bytes(raw)
        summary[kind] = {'count': len(bodies), 'sha256': hashlib.sha256(raw).hexdigest(),
                         'utf8_bytes': [len(body.encode()) for body in bodies],
                         'characters': [len(body) for body in bodies]}
    (args.output / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
