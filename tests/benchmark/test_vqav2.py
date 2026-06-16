import json
from typing import Dict, Optional

from evalscope.api.benchmark import DataAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ContentImage
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.vqav2.vqav2_adapter import _normalize_vqa_answer, _vqa_exact_match, _vqa_soft_accuracy
from evalscope.config import TaskConfig


def _adapter(name: str = 'vqav2', dataset_args: Optional[Dict] = None) -> DataAdapter:
    config = TaskConfig(datasets=[name], dataset_args={name: dataset_args or {}})
    return get_benchmark(name, config=config)


def test_vqav2_record_to_sample() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample({
        'question': 'What animal is shown?',
        'image': {'bytes': b'fake-image-bytes'},
        'answers': ['cat', 'cat', 'dog'],
        'multiple_choice_answer': 'cat',
        'question_id': 42,
    })

    assert isinstance(sample.input[0].content[1], ContentImage)
    assert sample.metadata['question'] == 'What animal is shown?'
    assert sample.metadata['answers'] == ['cat', 'cat', 'dog']
    assert json.loads(sample.target) == ['cat', 'cat', 'dog']


def test_vqav2_extract_answer() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample({
        'question': 'How many?',
        'image': {'bytes': b'img'},
        'answers': ['2'],
    })
    task_state = TaskState(
        model='mock',
        sample=sample,
        output=ModelOutput.from_content(model='mock', content='There are two.\nANSWER: 2'),
        completed=True,
    )

    extracted = adapter.extract_answer(task_state.output.completion, task_state)
    assert extracted == '2'


def test_vqav2_scoring() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample({
        'question': 'What animal?',
        'image': {'bytes': b'img'},
        'answers': ['cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'],
        'question_id': 7,
    })
    task_state = TaskState(
        model='mock',
        sample=sample,
        output=ModelOutput.from_content(model='mock', content='ANSWER: cat'),
        completed=True,
    )

    result = adapter.calculate_metrics(task_state)
    assert result.score.extracted_prediction == 'cat'
    assert abs(result.score.value['vqa_score'] - 2 / 3) < 1e-6
    assert result.score.value['exact_match'] == 1.0
    assert result.score.main_score_name == 'vqa_score'


def test_vqa_soft_accuracy_thresholds() -> None:
    assert _vqa_soft_accuracy('cat', ['cat', 'cat', 'cat', 'dog']) == 1.0
    assert _vqa_soft_accuracy('cat', ['cat', 'dog', 'dog', 'dog']) == abs(1 / 3)
    assert _vqa_soft_accuracy('bird', ['cat', 'dog']) == 0.0


def test_vqa_normalize_answer() -> None:
    assert _normalize_vqa_answer('A cat.') == 'cat'
    assert _normalize_vqa_answer('Three') == '3'
    assert _normalize_vqa_answer('the dog') == 'dog'


def test_vqa_exact_match() -> None:
    assert _vqa_exact_match('cat', ['cat', 'dog']) == 1.0
    assert _vqa_exact_match('bird', ['cat', 'dog']) == 0.0
