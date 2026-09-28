"""Replay fixed outputs through native adapters, scoring, aggregation and reports.

Run this same file with the old checkout on PYTHONPATH and its Python environment,
then with the migration environment. No model service or paid inference is used.
"""

import argparse
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from evalscope import TaskConfig, run_task
from evalscope.api.model import ModelAPI, ModelOutput
from evalscope.api.registry import get_benchmark, register_model_api


def fixed_prediction(answer: str, variant: str) -> str:
    """Create a known correct, incorrect, empty, or equivalent rendering."""
    answer = answer.strip().strip('$').strip()
    if variant == 'empty':
        return ''
    if variant == 'incorrect':
        try:
            if not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', answer):
                raise InvalidOperation
            number = Decimal(answer)
            wrong = str(number * 100) if number.is_finite() and number else '1'
        except InvalidOperation:
            wrong = '987654321098765432109876543210'
        return f'\\boxed{{{wrong}}}'
    if variant == 'equivalent':
        try:
            if not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', answer):
                raise InvalidOperation
            number = Decimal(answer)
            if number.is_finite():
                answer = f'{number}+0'
        except InvalidOperation:
            answer = re.sub(
                r'\\frac\{(\d+)\}\{(\d+)\}',
                lambda m: rf'\frac{{{2 * int(m[1])}}}{{{2 * int(m[2])}}}',
                answer,
            )
        return f'Final answer: $\\displaystyle {answer}$'
    return f'\\boxed{{{answer}}}'


class ReplayModel(ModelAPI):
    """Use the exact native prompt as a key, independent of thread scheduling."""

    def __init__(self, *args: Any, replay_file: str, benchmark: str, variant: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        records = json.loads(Path(replay_file).read_text())
        adapter = get_benchmark(
            benchmark, config=TaskConfig(datasets=[benchmark], dataset_args={benchmark: {'few_shot_num': 0}})
        )
        self.outputs = {}
        for record in records:
            sample = adapter.record_to_sample(record)
            prompt = adapter.format_prompt_template(sample) if isinstance(sample.input, str) else sample.input[-1].text
            self.outputs[prompt.strip()] = record.get(
                '_historical_prediction', fixed_prediction(sample.target, variant)
            )

    def generate(self, input: list, tools: list, tool_choice: Any, config: Any) -> ModelOutput:
        """Never issue an external request; unknown prompts fail instead of guessing."""
        return ModelOutput.from_content(self.model_name, self.outputs[input[-1].text.strip()])


@register_model_api(name='mock_llm_math_replay')
def replay_model() -> type[ModelAPI]:
    """Expose an offline validation model only while this script is running."""
    return ReplayModel


def run_replay(args: argparse.Namespace) -> None:
    """Evaluate every record in the frozen JSON files with real pipeline components."""
    args.output.mkdir(parents=True, exist_ok=True)
    variants = ['correct', 'incorrect', 'empty', 'equivalent'] if args.variant == 'all' else [args.variant]
    for benchmark in args.benchmarks:
        source = args.data / f'{benchmark}.json'
        records = json.loads(source.read_text())
        if args.limit:
            records = records[: args.limit]
        dataset_dir = args.output / 'source' / benchmark
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'test.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        replay_file = dataset_dir / 'replay.json'
        replay_file.write_text(json.dumps(records, ensure_ascii=False))
        for variant in variants:
            folder = args.output / benchmark / variant
            run_task(
                TaskConfig(
                    model=f'math-replay-{variant}',
                    eval_type='mock_llm_math_replay',
                    datasets=[benchmark],
                    dataset_args={
                        benchmark: {
                            'local_path': str(dataset_dir.resolve()),
                            'subset_list': (
                                ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
                                if benchmark == 'math_500'
                                else ['default']
                            ),
                            'few_shot_num': 0,
                        }
                    },
                    judge={'strategy': 'rule'},
                    eval_batch_size=4,
                    work_dir=str(folder),
                    no_timestamp=True,
                    model_args={'replay_file': str(replay_file.resolve()), 'benchmark': benchmark, 'variant': variant},
                )
            )
    collect_results(args.output)


def collect_results(root: Path) -> None:
    """Persist extracted answers, scores, status, identities and aggregate denominators."""
    rows = []
    aggregates = {}
    for path in sorted(root.glob('*/*/reviews/*/*.jsonl')):
        benchmark, variant = path.relative_to(root).parts[:2]
        for line in path.read_text().splitlines():
            item = json.loads(line)
            score = item['sample_score']['score']
            rows.append(
                {
                    'benchmark': benchmark,
                    'variant': variant,
                    'subset': path.stem,
                    'index': item['index'],
                    'target': item['target'],
                    'score': score,
                    'sample_metadata': item['sample_score'].get('sample_metadata', {}),
                }
            )
    for path in sorted(root.glob('*/*/reports/*/*.json')):
        aggregates[str(path.relative_to(root))] = json.loads(path.read_text())
    (root / 'samples.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    (root / 'aggregates.json').write_text(json.dumps(aggregates, ensure_ascii=False, indent=2) + '\n')
    print(f'Collected {len(rows)} reviews under {root}')


def compare_results(args: argparse.Namespace) -> None:
    """Compare identical sample identities and preserve every score/extraction change."""

    def read(root: Path) -> dict:
        return {
            (r['benchmark'], r['variant'], r['subset'], r['index']): r
            for r in map(json.loads, (root / 'samples.jsonl').read_text().splitlines())
        }

    old, new = read(args.old), read(args.output)
    if old.keys() != new.keys():
        raise ValueError('Old and new sample identities differ')
    differences = []
    for key, previous in old.items():
        current = new[key]
        if previous['target'] != current['target']:
            raise ValueError(f'Reference drift for {key}')
        if previous['score'] != current['score']:
            differences.append({'identity': key, 'old': previous, 'new': current})
    (args.output / 'differences.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in differences)
    )
    print(f'{len(differences)} changed reviews; inspect differences.jsonl')


def freeze_data(args: argparse.Namespace) -> None:
    """Download once and freeze source records by SHA256 before either implementation runs."""
    from evalscope.api.dataset.hub import DatasetHub

    args.data.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for benchmark, dataset_id, subset, count in [
        ('gsm8k', 'AI-ModelScope/gsm8k', 'main', 1319),
        ('math_500', 'AI-ModelScope/MATH-500', 'default', 500),
    ]:
        path = args.data / f'{benchmark}.json'
        if not path.exists():
            dataset = DatasetHub(dataset_id).load(split='test', subset=subset)
            path.write_text(json.dumps(list(dataset), ensure_ascii=False) + '\n')
        records = json.loads(path.read_text())
        assert len(records) == count
        manifest[benchmark] = {
            'dataset_id': dataset_id,
            'samples': count,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.data / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def freeze_history(args: argparse.Namespace) -> None:
    """Recover real historical predictions by question identity, never by index alone."""
    records = json.loads((args.data / 'gsm8k.json').read_text())
    manifest = []
    for run in args.historical_runs:
        recovered = []
        for review_file in run.glob('reviews/*/gsm8k*.jsonl'):
            reviews = {r['index']: r for r in map(json.loads, review_file.read_text().splitlines())}
            prediction_file = run / 'predictions' / review_file.parent.name / review_file.name
            for prediction in map(json.loads, prediction_file.read_text().splitlines()):
                review = reviews[prediction['index']]
                prompt = '\n'.join(m['content'] for m in prediction['messages'] if m.get('role') == 'user')
                candidates = [record for record in records if record['question'] in prompt]
                if len(candidates) != 1:
                    raise ValueError(f'Ambiguous historical question at {prediction_file}:{prediction["index"]}')
                record = dict(candidates[0])
                target = review['target']
                if isinstance(target, list):
                    target = target[0]
                assert record['answer'].split('####')[-1].strip() == target
                record['_historical_prediction'] = prediction['model_output']['choices'][0]['message']['content']
                recovered.append(record)
        destination = args.output / run.name
        destination.mkdir(parents=True, exist_ok=True)
        source = destination / 'gsm8k.json'
        source.write_text(json.dumps(recovered, ensure_ascii=False) + '\n')
        manifest.append(
            {'run': str(run), 'samples': len(recovered), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        )
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main() -> None:
    """Run preparation, replay or the old/new comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run', 'compare', 'collect', 'history'])
    parser.add_argument('--data', type=Path, default=Path('validation/math_verify/data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old', type=Path)
    parser.add_argument('--benchmarks', nargs='+', default=['gsm8k', 'math_500'])
    parser.add_argument(
        '--variant', choices=['all', 'correct', 'incorrect', 'empty', 'equivalent', 'historical'], default='all'
    )
    parser.add_argument('--limit', type=int)
    parser.add_argument('--historical-runs', nargs='+', type=Path)
    args = parser.parse_args()
    if args.action == 'run':
        run_replay(args)
    elif args.action == 'compare':
        compare_results(args)
    elif args.action == 'freeze':
        freeze_data(args)
    elif args.action == 'history':
        freeze_history(args)
    else:
        collect_results(args.output)


if __name__ == '__main__':
    main()
