# Copyright (c) Alibaba, Inc. and its affiliates.
"""ASR benchmarks report corpus WER (total errors / total reference words), as lmms-eval and Qwen-Audio do."""
from pathlib import Path

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.report import Report

# One error in a 2-word utterance and a perfect 18-word utterance: corpus WER is 1 / 20, not mean(1/2, 0).
UTTERANCES = [
    ('thank you', 'thank yo'),
    (
        'he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces',
        'he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces',
    ),
]
CASES = {
    'librispeech': ({}, 'v1.3'),
    'fleurs': ({'lang_id': 'en'}, 'v1.2'),
    'common_voice_15': ({'locale': 'en'}, 'v1.2'),
}


@pytest.mark.parametrize('name', CASES)
@pytest.mark.parametrize(
    ('utterances', 'expected'),
    [
        (UTTERANCES, 1 / 20),
        ([('hello world', 'hello world'), ('', 'extra')], 0.5),
        ([('hello world', 'hello world'), ('uh', 'extra')], 0.5),
        ([('hello world', 'hello world'), ('...', 'extra')], 0.5),
        ([('', 'extra words'), ('', 'more')], 3.0),
        ([('', ''), ('', '')], 0.0),
    ],
    ids=['ordinary', 'silent-insertion', 'normalized-empty', 'punctuation', 'only-insertions', 'silence'],
)
def test_corpus_wer(name: str, utterances: list[tuple[str, str]], expected: float, tmp_path: Path) -> None:
    metadata, version = CASES[name]
    adapter = get_benchmark(name, TaskConfig(datasets=[name]))
    scores = []
    for index, (reference, hypothesis) in enumerate(utterances):
        state = TaskState(
            model='mock',
            sample=Sample(id=index, input='q', target=reference, metadata=metadata),
            output=ModelOutput.from_content('mock', hypothesis),
            completed=True,
        )
        scores.append(adapter.calculate_metrics(state))

    (wer,) = adapter.aggregate_scores(scores)
    assert wer.metric_name == 'wer'
    assert wer.score == pytest.approx(expected)
    assert wer.ids == [0, 1]

    # A silent subset can contribute insertions even when all reference words are in another subset.
    partitions = {f'subset-{i}': adapter.aggregate_scores([score]) for i, score in enumerate(scores)}
    report = adapter.generate_report(partitions, 'mock', str(tmp_path))
    assert report.score == pytest.approx(expected)
    assert report.metrics[0].categories[0].score == pytest.approx(expected)
    assert Report.from_dict(report.to_dict()).metrics == report.metrics
    assert adapter.benchmark_meta.evaluation_version == version
