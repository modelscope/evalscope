"""Freeze real scoring contexts and replay every affected adapter without model inference."""

import argparse
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from evalscope import TaskConfig
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.constants import JudgeScoreType
from evalscope.metrics.judge.llm_judge import DEFAULT_PROMPT_TEMPLATE, LLMJudge

ROOT = Path(__file__).resolve().parents[2]


class FixedJudge:
    """Offline protocol double: half of each official criterion, or a false answer verdict."""

    score_type = JudgeScoreType.PATTERN
    score_mapping = {'A': 1.0, 'B': 0.0}
    prompt_template = DEFAULT_PROMPT_TEMPLATE
    system_prompt = None
    judge_id = model_id = 'offline-fixture'
    build_prompt = LLMJudge.build_prompt

    def generate(self, messages: list) -> ModelOutput:
        from evalscope.benchmarks.hipho.utils import criterion_points

        prompt = messages[-1].text
        if 'Grading Criterion:' in prompt:
            criterion = prompt.split('Grading Criterion:', 1)[1].split('Instructions:', 1)[0]
            reply = {'awarded': criterion_points(criterion) / 2}
        else:
            reply = {'correct': False}
        return ModelOutput.from_content(self.model_id, json.dumps(reply))


def sample_context(sample: Sample) -> dict:
    """Freeze score inputs; figure bytes are excluded because this replay performs no inference."""
    return {
        'input': sample.input if isinstance(sample.input, str) else '\n'.join(m.text for m in sample.input),
        'target': sample.target,
        'metadata': sample.metadata or {},
        'choices': sample.choices or [],
    }


def freeze(args: argparse.Namespace) -> None:
    """Retain dataset targets/metadata and select actual numeric, symbolic and judge branches."""
    cases = []
    for name in args.benchmarks:
        path = ROOT / f'evalscope/benchmarks/_meta/{name}.json'
        entry = json.loads(path.read_text())
        data = entry['sample_example']['data']
        if name == 'hipho':
            continue
        content = data['input'][0]['content']
        if isinstance(content, list):
            content = '\n'.join(part.get('text', '') for part in content)
        sample = {
            'input': content,
            'target': data['target'],
            'metadata': data.get('metadata', {}),
            'choices': data.get('choices', []),
        }
        prediction = f'\\boxed{{{data["target"].strip().strip("$")}}}'
        if sample['choices']:
            prediction = f'答案：{data["target"]}' if name == 'cmmu' else f'ANSWER: {data["target"]}'
        if name == 'chartqa':
            prediction = f'ANSWER: {data["target"]}'
        elif name == 'measure_bench':
            config = json.loads(sample['metadata']['evaluator_kwargs'])
            interval = config.get('interval') or config['intervals'][0]
            units = config.get('units', [])
            if units and isinstance(units[0], list):
                units = units[0]
            prediction = f'Answer: {interval[0]} {units[0] if units else ""}'.strip()
        cases.append(
            {
                'benchmark': name,
                'branch': 'cached_real_score_context',
                'sample': sample,
                'prediction': prediction,
                'expected': 1.0,
                'source': str(path.relative_to(ROOT)),
                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'prompt_truncated': entry['sample_example'].get('truncated', False),
            }
        )

    for path in args.data.glob('olympiad_*.json'):
        records = json.loads(path.read_text())
        selected = {}
        for record in records:
            key = (
                record['answer_type'],
                record['is_multiple_answer'],
                bool(record['error']),
                ',' in (record['error'] or ''),
            )
            selected.setdefault(key, record)
        adapter = get_benchmark('olympiad_bench')
        for key, record in selected.items():
            sample = adapter.record_to_sample(record)
            answer = record['final_answer'][0].replace('$', '').strip().rstrip('.')
            cases.append(
                {
                    'benchmark': 'olympiad_bench',
                    'branch': str(key),
                    'sample': sample_context(sample),
                    'prediction': f'\\boxed{{{answer}}}',
                    'expected': 1.0,
                    'source': path.name,
                    'source_id': record['id'],
                }
            )
            if record['answer_type'] == 'Numerical' and not record['is_multiple_answer']:
                try:
                    number = Decimal(answer)
                except InvalidOperation:
                    continue
                if number and number.is_finite():
                    cases.append(
                        {
                            'benchmark': 'olympiad_bench',
                            'branch': 'wrong_scale',
                            'sample': sample_context(sample),
                            'prediction': f'\\boxed{{{number * 100}}}',
                            'expected': 0.0,
                            'source': path.name,
                            'source_id': record['id'],
                        }
                    )

    # Real DocMath records from the cached public test split include zero, negative and large values.
    from datasets import Dataset

    cache = args.cache / 'yale-nlp___doc_math-eval'
    source = next(cache.rglob('doc_math-eval-simpshort_test.arrow'))
    records = list(Dataset.from_file(str(source)))
    adapter = get_benchmark('docmath')
    selected = [records[0], next(r for r in records if r['ground_truth'] == 0)]
    for record in selected:
        sample = sample_context(adapter.record_to_sample(record))
        gold = str(record['ground_truth'])
        for branch, prediction, expected in [
            ('correct', gold, 1.0),
            ('empty', '', 0.0),
            ('parse_failure', 'no answer', 0.0),
            ('wrong_scale', str(Decimal(gold) * 100) if Decimal(gold) else '1', 0.0),
        ]:
            cases.append(
                {
                    'benchmark': 'docmath',
                    'branch': branch,
                    'sample': sample,
                    'prediction': prediction,
                    'expected': expected,
                    'source': source.name,
                    'source_id': record['question_id'],
                }
            )

    snapshot = args.cache / 'datasets/evalscope--HiPhO/snapshots/master/data'
    adapter = get_benchmark('hipho')
    chosen = {}
    for path in snapshot.glob('*.json'):
        for record in json.loads(path.read_text()):
            if 'question' not in record:
                continue
            if record.get('marking'):
                branch = 'official_marking'
            elif len(record.get('answer') or []) > 1:
                branch = 'multiple_answers'
            elif record.get('answer'):
                branch = 'single_answer'
            else:
                continue
            chosen.setdefault(branch, (path, record))
    for branch, (path, original) in chosen.items():
        record = {**original, 'image_question': []}
        sample = sample_context(adapter.record_to_sample(record))
        answer = ', '.join(record.get('answer') or [])
        cases.append(
            {
                'benchmark': 'hipho',
                'branch': branch,
                'sample': sample,
                'prediction': answer,
                'expected': 0.5 if branch == 'official_marking' else 1.0,
                'source': path.name,
                'source_id': record['id'],
            }
        )
        if branch == 'multiple_answers':
            count = len(record['answer'])
            cases.append(
                {
                    'benchmark': 'hipho',
                    'branch': 'partial_answers',
                    'sample': sample,
                    'prediction': ', '.join(record['answer'][:-1]),
                    'expected': (count - 1) / count,
                    'source': path.name,
                    'source_id': record['id'],
                }
            )
    args.fixture.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
    print(f'Frozen {len(cases)} real scoring contexts')


def run(args: argparse.Namespace) -> None:
    """Use normal adapter scoring and aggregators and record failures explicitly."""
    results = []
    grouped = {}
    for index, case in enumerate(json.loads(args.fixture.read_text())):
        name = case['benchmark']
        config = TaskConfig(datasets=[name], judge={'strategy': 'llm' if name == 'hipho' else 'rule'})
        adapter = get_benchmark(name, config)
        if name == 'hipho':
            adapter.llm_judge = FixedJudge()
        sample = Sample(id=index, **case['sample'])
        task_state = TaskState(
            model='offline',
            sample=sample,
            output=ModelOutput.from_content('offline', case['prediction']),
            completed=True,
        )
        try:
            score = adapter.calculate_metrics(task_state)
            result = {'case': case, 'score': score.model_dump(mode='json')}
            grouped.setdefault(name, (adapter, []))[1].append(score)
        except Exception as exc:
            result = {'case': case, 'error': f'{type(exc).__name__}: {exc}'}
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    aggregation = {
        name: [score.model_dump(mode='json') for score in adapter.aggregate_scores(scores)]
        for name, (adapter, scores) in grouped.items()
    }
    args.output.with_suffix('.aggregates.json').write_text(json.dumps(aggregation, ensure_ascii=False, indent=2) + '\n')
    errors = []
    for result in results:
        values = result.get('score', {}).get('score', {}).get('value', {})
        value = next(iter(values.values()), -1)
        result['matches_expected'] = 'error' not in result and abs(value - result['case']['expected']) < 1e-8
        if not result['matches_expected']:
            errors.append(result)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(results)} cases; {len(errors)} differ from expected')
    for result in errors:
        print(result['case']['benchmark'], result['case']['branch'], result.get('error', result.get('score')))


def main() -> None:
    """Freeze once, then run the same fixture in old and new isolated environments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--benchmarks', nargs='+', help='Benchmark names to freeze')
    parser.add_argument('--data', type=Path, default=ROOT / 'validation/math_verify/data')
    parser.add_argument('--fixture', type=Path, default=ROOT / 'validation/math_verify/adapter_fixtures.json')
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/modelscope/hub/datasets')
    parser.add_argument('--output', type=Path, default=ROOT / 'validation/math_verify/adapter_results.json')
    args = parser.parse_args()
    if args.action == 'freeze' and not args.benchmarks:
        parser.error('--benchmarks is required when freezing scoring contexts')
    freeze(args) if args.action == 'freeze' else run(args)


if __name__ == '__main__':
    main()
