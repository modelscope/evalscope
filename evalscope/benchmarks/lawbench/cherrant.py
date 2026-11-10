# Copyright (c) Alibaba, Inc. and its affiliates.
# Input preparation follows OpenCompass/LawBench (Apache-2.0).
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List


def validate_toolkit(toolkit: Path) -> Path:
    """Resolve the official LawBench evaluation/utils directory without importing it."""
    toolkit = toolkit.expanduser().resolve()
    for script in ('parallel_to_m2.py', 'compare_m2_for_evaluation.py'):
        if not (toolkit / script).is_file():
            raise FileNotFoundError(
                f'ChERRANT requires {toolkit / script}; set extra_params.cherrant_path '
                'to the official LawBench evaluation/utils directory.'
            )
    return toolkit


def score_corrections(rows: List[Dict[str, str]], toolkit: Path, python: str, timeout: float) -> float:
    """Run the official character-level corpus F0.5 with private temporary files."""
    toolkit = validate_toolkit(toolkit)
    if not rows or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('ChERRANT requires nonempty rows and a positive finite timeout.')
    predictions, references = [], []
    for index, row in enumerate(rows):
        prompt, prediction, reference = row['origin_prompt'], row['prediction'], row['refr']
        start = prompt.index('句子：\n') + 4
        source = re.sub(r'\n|\t', '', prompt[start:].split('\n')[0])
        prediction = re.sub(r'\n|\t', '', prediction)
        if len(prediction) - len(reference) > 5:
            prediction = prediction[: len(reference) + 5]
        prediction = prediction or '无内容'
        reference = re.sub(r'\n|\t', '', reference)
        predictions.append(f'{index} \t {source} \t {prediction} \n')
        references.append(f'{index} \t {source} \t {reference} \n')

    env = {**os.environ, 'KMP_DUPLICATE_LIB_OK': 'True'}
    with tempfile.TemporaryDirectory(prefix='evalscope-lawbench-') as directory:
        work = Path(directory)
        (work / 'pred.para').write_text(''.join(predictions), encoding='utf-8')
        (work / 'gold.para').write_text(''.join(references), encoding='utf-8')
        commands = [
            [python, str(toolkit / 'parallel_to_m2.py'), '-f', name, '-o', f'{name}.m2', '-g', 'char']
            for name in ('pred.para', 'gold.para')
        ]
        commands.append(
            [
                python,
                str(toolkit / 'compare_m2_for_evaluation.py'),
                '-hyp',
                'pred.para.m2',
                '-ref',
                'gold.para.m2',
            ]
        )
        output = ''
        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    check=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f'ChERRANT timed out after {timeout} seconds per subprocess.') from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f'ChERRANT failed: {exc.stderr[-2000:]}') from exc
            output = result.stdout

    lines = output.splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.split('\t') == ['TP', 'FP', 'FN', 'Prec', 'Rec', 'F0.5']:
            fields = lines[index + 1].split('\t')
            if len(fields) == 6:
                value = float(fields[-1])
                if math.isfinite(value) and 0 <= value <= 1:
                    return value
    raise ValueError('ChERRANT did not produce a valid corpus F0.5 score.')
