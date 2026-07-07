import importlib
import sys
import types

from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
from evalscope.config import TaskConfig


ROLE_TASK = 'perspective_gap_role_assignment'
PROMPT_TASK = 'perspective_gap_prompt_writing'


def import_adapter_module():
    return importlib.import_module('evalscope.benchmarks.perspective_gap.perspective_gap_adapter')


def task_config():
    return TaskConfig(datasets=[ROLE_TASK, PROMPT_TASK], dataset_args={})


def sample_record():
    return {
        'evaluation_id': 'pg_unit__seed_1',
        'scenario_id': 'pg_unit',
        'shuffle_seed': 1,
        'roles': ['coder', 'reviewer'],
        'fragments': [
            {'id': 'f1', 'text': 'Coder-only instructions.', 'is_distractor': False},
            {'id': 'f2', 'text': 'Reviewer-only instructions.', 'is_distractor': False},
            {'id': 'f3', 'text': 'Distractor instructions.', 'is_distractor': True},
        ],
        'distractor_id': 'f3',
        'reference_need_sets': {'coder': ['f1'], 'reviewer': ['f2']},
        'role_assignment_prompt': 'Return JSON fragment IDs per role.',
        'prompt_writing_prompt': 'Write one markdown prompt per role.',
    }


def install_fake_scoring(monkeypatch, calls):
    package = types.ModuleType('perspective_gap')
    scoring = types.ModuleType('perspective_gap.scoring')

    def score_role_assignment(response, reference_need_sets, distractor_id=None):
        calls.append(('role_assignment', response, reference_need_sets, distractor_id))
        return {
            'pass': True,
            'metrics': {'strict_pass': 1.0, 'net_match_score': 1.0},
            'counts': {'tp': 2, 'fp': 0, 'fn': 0, 'distractor_leak': 0},
        }

    def score_prompt_writing(response, fragments, reference_need_sets, distractor_id=None):
        calls.append(('prompt_writing', response, fragments, reference_need_sets, distractor_id))
        return {
            'pass': False,
            'metrics': {'strict_pass': 0.0, 'net_match_score': 0.5},
            'counts': {'tp': 1, 'fp': 0, 'fn': 1, 'distractor_leak': 0},
        }

    scoring.score_role_assignment = score_role_assignment
    scoring.score_prompt_writing = score_prompt_writing
    package.scoring = scoring
    monkeypatch.setitem(sys.modules, 'perspective_gap', package)
    monkeypatch.setitem(sys.modules, 'perspective_gap.scoring', scoring)


def test_perspective_gap_tasks_register_without_importing_scorer(monkeypatch):
    monkeypatch.delitem(sys.modules, 'perspective_gap.scoring', raising=False)

    import_adapter_module()

    assert 'perspective_gap.scoring' not in sys.modules
    assert ROLE_TASK in BENCHMARK_REGISTRY
    assert PROMPT_TASK in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY[ROLE_TASK].data_adapter is not BENCHMARK_REGISTRY[PROMPT_TASK].data_adapter


def test_role_assignment_sample_uses_role_prompt_and_metadata():
    import_adapter_module()
    adapter = get_benchmark(ROLE_TASK, config=task_config())

    sample = adapter.record_to_sample(sample_record())

    assert sample.input == 'Return JSON fragment IDs per role.'
    assert sample.target == '{"coder": ["f1"], "reviewer": ["f2"]}'
    assert sample.metadata['task'] == 'role_assignment'
    assert sample.metadata['evaluation_id'] == 'pg_unit__seed_1'
    assert sample.metadata['reference_need_sets'] == {'coder': ['f1'], 'reviewer': ['f2']}
    assert sample.metadata['distractor_id'] == 'f3'


def test_prompt_writing_sample_uses_prompt_writing_prompt_and_fragments():
    import_adapter_module()
    adapter = get_benchmark(PROMPT_TASK, config=task_config())

    sample = adapter.record_to_sample(sample_record())

    assert sample.input == 'Write one markdown prompt per role.'
    assert sample.target == '{"coder": ["f1"], "reviewer": ["f2"]}'
    assert sample.metadata['task'] == 'prompt_writing'
    assert sample.metadata['fragments'][0]['id'] == 'f1'
    assert sample.metadata['distractor_id'] == 'f3'


def test_role_assignment_scoring_calls_perspective_gap_scoring(monkeypatch):
    calls = []
    install_fake_scoring(monkeypatch, calls)
    import_adapter_module()
    adapter = get_benchmark(ROLE_TASK, config=task_config())
    sample = adapter.record_to_sample(sample_record())
    task_state = TaskState(
        model='unit',
        sample=sample,
        output=ModelOutput.from_content(model='unit', content='{"coder": ["f1"], "reviewer": ["f2"]}'),
        completed=True,
    )

    score = adapter.match_score(
        original_prediction='raw role output',
        filtered_prediction='filtered role output',
        reference=sample.target,
        task_state=task_state,
    )

    assert calls == [('role_assignment', 'filtered role output', {'coder': ['f1'], 'reviewer': ['f2']}, 'f3')]
    assert score.value['strict_pass'] == 1.0
    assert score.main_score_name == 'strict_pass'
    assert score.metadata['counts']['tp'] == 2


def test_prompt_writing_scoring_calls_perspective_gap_scoring(monkeypatch):
    calls = []
    install_fake_scoring(monkeypatch, calls)
    import_adapter_module()
    adapter = get_benchmark(PROMPT_TASK, config=task_config())
    sample = adapter.record_to_sample(sample_record())
    task_state = TaskState(
        model='unit',
        sample=sample,
        output=ModelOutput.from_content(model='unit', content='# coder\nCoder-only instructions.'),
        completed=True,
    )

    score = adapter.match_score(
        original_prediction='raw prompt output',
        filtered_prediction='filtered prompt output',
        reference=sample.target,
        task_state=task_state,
    )

    assert calls == [(
        'prompt_writing',
        'filtered prompt output',
        sample_record()['fragments'],
        {'coder': ['f1'], 'reviewer': ['f2']},
        'f3',
    )]
    assert score.value['strict_pass'] == 0.0
    assert score.metadata['metrics']['net_match_score'] == 0.5
