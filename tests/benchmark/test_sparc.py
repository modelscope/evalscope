import copy
import hashlib
from typing import Any, Dict, List

import pytest

import evalscope  # noqa: F401  # imported for benchmark and metric registration side effects
from evalscope.api.benchmark import DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.sparc.utils import SYSTEM_PROMPT, get_prompt
from evalscope.config import TaskConfig
from evalscope.constants import HubType

# Test puzzle c6739963a7308111 from evalscope/SPaRC (config `all`, split `test`), without the unused
# `text_visualization` field. S is at (0,4), E at (4,0); the puzzle has four stored solutions.
PUZZLE: Dict[str, Any] = {
    'id': 'c6739963a7308111',
    'difficulty_level': 3,
    'difficulty_score': 2.027591071174611,
    'grid_size': {'height': 2, 'width': 2},
    'polyshapes': '{"16":[[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],"784":[[0,1,1,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]}',
    'puzzle_array': [
        ['+', '+', '+', '+', 'E'],
        ['+', 'o-P', '+', 'P-W-784', '+'],
        ['+', '+', '+', '+', '+'],
        ['+', 'P-R-16', '+', 'o-P', '+'],
        ['S', '+', '+', '+', '+'],
    ],
    'solution_count': 4,
    'solutions': [
        {
            'index': index,
            'path': [{'x': x, 'y': y} for x, y in path],
            'pathLength': len(path),
        }
        for index, path in enumerate(
            [
                [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (4, 3), (4, 2), (4, 1), (4, 0)],
                [(0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
                [(0, 4), (0, 3), (0, 2), (1, 2), (2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (4, 3), (4, 2), (4, 1), (4, 0)],
                [(0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            ]
        )
    ],
}

# SHA-256 of the user prompt that the reference package (sparc.prompts.single_shot.get_prompt) builds
# for PUZZLE, so any change to the copied prompt builder fails this test.
REFERENCE_PROMPT_SHA256 = '320f7b8028116fbb5b93f79a11fedab9f76d2ddc54ad222b604903d1400fd69e'


def _adapter(**config: Any) -> DefaultDataAdapter:
    return get_benchmark('sparc', TaskConfig(datasets=['sparc'], **config))


def _path_text(path: List[Dict[str, int]]) -> str:
    return ', '.join(f'({p["x"]},{p["y"]})' for p in path)


def _score(adapter: DefaultDataAdapter, sample: Sample, completion: str) -> Dict[str, float]:
    state = TaskState(
        model='mock',
        sample=sample,
        messages=[],
        output=ModelOutput.from_content(model='mock', content=completion),
        completed=True,
    )
    return adapter.calculate_metrics(state).score.value


def test_sparc_registered_with_level_subsets_and_pinned_revision() -> None:
    adapter = _adapter()

    assert adapter.dataset_id == 'evalscope/SPaRC'
    assert adapter.dataset_revision == '104103a0abae909a46b85f11fb2f949fc7448f87'
    assert adapter.default_subset == 'all'
    assert adapter.subset_list == ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
    assert adapter.eval_split == 'test'
    assert adapter.few_shot_num == 0
    assert adapter.metric_list == ['accuracy', 'invalid_path_rate']
    assert adapter.primary_metric.name == 'accuracy'
    assert adapter.system_prompt == SYSTEM_PROMPT


def test_record_to_sample_uses_reference_prompt() -> None:
    adapter = _adapter()

    sample = adapter.record_to_sample(copy.deepcopy(PUZZLE))
    messages = adapter.process_sample_str_input(sample, subset=sample.subset_key)

    assert sample.subset_key == 'Level 3'
    assert sample.target == _path_text(PUZZLE['solutions'][0]['path'])
    assert sample.metadata['puzzle']['solutions'] == PUZZLE['solutions']
    assert isinstance(messages[0], ChatMessageSystem)
    assert messages[0].content == 'You are an expert at solving puzzles games.'
    assert isinstance(messages[1], ChatMessageUser)
    # get_prompt returns only the user message; the system prompt comes from BenchmarkMeta.
    assert isinstance(get_prompt(PUZZLE), str)
    assert messages[1].content == get_prompt(PUZZLE)
    assert hashlib.sha256(messages[1].content.encode()).hexdigest() == REFERENCE_PROMPT_SHA256


@pytest.mark.parametrize('index', range(4))
def test_every_stored_solution_is_correct(index: int) -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(copy.deepcopy(PUZZLE))
    path = _path_text(PUZZLE['solutions'][index]['path'])

    value = _score(adapter, sample, f'The line goes around the squares.\n\n####\n{path}')

    assert value == {'accuracy': 1.0, 'invalid_path_rate': 0.0}


@pytest.mark.parametrize(
    'completion, expected_invalid',
    [
        # A valid line from S to E that breaks the rules: wrong, but not an invalid path.
        ('#### (0,4), (1,4), (2,4), (2,3), (2,2), (3,2), (4,2), (4,1), (4,0)', 0.0),
        # Steps onto the rule cell (1,3).
        ('#### (0,4), (0,3), (1,3), (2,3), (2,2), (2,1), (2,0), (3,0), (4,0)', 1.0),
        # Not connected: jumps from (0,4) to (2,4).
        ('#### (0,4), (2,4), (3,4), (4,4), (4,3), (4,2), (4,1), (4,0)', 1.0),
        # Stops before the end node.
        ('#### (0,4), (0,3), (0,2), (0,1), (0,0)', 1.0),
        # Starts at the end node, so the parser stops after one node.
        ('#### (4,0), (3,0), (2,0), (1,0), (0,0), (0,1), (0,2), (0,3), (0,4)', 1.0),
        ('I could not find a path.', 1.0),
        ('', 1.0),
    ],
)
def test_wrong_answers_score_zero(completion: str, expected_invalid: float) -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(copy.deepcopy(PUZZLE))

    value = _score(adapter, sample, completion)

    assert value == {'accuracy': 0.0, 'invalid_path_rate': expected_invalid}


def test_only_text_after_last_marker_is_parsed() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(copy.deepcopy(PUZZLE))
    solution = _path_text(PUZZLE['solutions'][3]['path'])

    correct_last = _score(adapter, sample, f'#### (0,4), (1,4)\nLet me fix that.\n#### {solution}')
    wrong_last = _score(adapter, sample, f'#### {solution}\nLet me change it.\n#### (0,4), (1,4)')

    assert correct_last['accuracy'] == 1.0
    assert wrong_last['accuracy'] == 0.0


def test_aggregation_reports_accuracy_and_invalid_path_rate() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(copy.deepcopy(PUZZLE))
    completions = [
        '#### ' + _path_text(PUZZLE['solutions'][0]['path']),
        '#### (0,4), (1,4), (2,4), (2,3), (2,2), (3,2), (4,2), (4,1), (4,0)',
        'no answer',
        '',
    ]
    scores = []
    for i, completion in enumerate(completions):
        state = TaskState(
            model='mock',
            sample=sample,
            output=ModelOutput.from_content(model='mock', content=completion),
            completed=True,
        )
        sample_score = adapter.calculate_metrics(state)
        scores.append(SampleScore(score=sample_score.score, sample_id=i))

    aggregates = {agg.metric_name: agg for agg in adapter.aggregate_scores(scores)}

    assert aggregates['accuracy'].score == pytest.approx(0.25)
    assert aggregates['invalid_path_rate'].score == pytest.approx(0.5)
    assert aggregates['accuracy'].num == 4


@pytest.mark.parametrize('configured_hub', [HubType.MODELSCOPE, HubType.HUGGINGFACE])
def test_default_hub_is_modelscope_without_hf_override(configured_hub: str) -> None:
    adapter = _adapter(dataset_hub=configured_hub)

    assert adapter.dataset_hub == configured_hub
    # SPaRC ships on ModelScope, so the adapter must not override load() to redirect the hub.
    assert type(adapter).load is DefaultDataAdapter.load
