import json
from typing import List

from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.mt_bench.mt_bench_adapter import MTBenchAdapter
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus


class TwoTurnModel:
    def __init__(self, replies: List[str]) -> None:
        self.config = GenerateConfig()
        self.replies = replies
        self.calls = []

    def generate(self, input, config=None):
        self.calls.append((input, config))
        return ModelOutput.from_content('two-turn-model', self.replies[len(self.calls) - 1])


class ScriptedJudge:
    def __init__(self, replies: List[str]) -> None:
        self.replies = replies
        self.calls = []
        self.judge_id = self.model_id = 'scripted-judge'

    def generate(self, messages):
        self.calls.append(messages)
        return ModelOutput.from_content('scripted-judge', self.replies[len(self.calls) - 1])


def make_adapter() -> MTBenchAdapter:
    config = TaskConfig(
        model='model',
        datasets=['mt_bench'],
        judge={'strategy': 'llm', 'models': [{'model_id': 'judge'}]},
    )
    return get_benchmark('mt_bench', config)


def make_state(adapter: MTBenchAdapter, category: str = 'writing') -> TaskState:
    record = {
        'category': category,
        'prompt': ['First question', 'Second question'],
        'reference': ['First reference', 'Second reference'] if category in {'reasoning', 'math', 'coding'} else [],
        'prompt_id': 1,
    }
    sample = adapter.record_to_sample(record)
    sample.id = 0
    messages = [
        ChatMessageUser(content='First question'),
        ChatMessageAssistant(content='First answer'),
        ChatMessageUser(content='Second question'),
        ChatMessageAssistant(content='Second answer'),
    ]
    return TaskState(
        model='model',
        sample=sample,
        messages=messages,
        output=ModelOutput.from_content('model', 'Second answer'),
        completed=True,
    )


def test_record_to_sample_uses_category_as_subset() -> None:
    adapter = make_adapter()
    sample = adapter.record_to_sample(
        {
            'category': 'writing',
            'prompt': ['Draft a message.', 'Rewrite it as a poem.'],
            'reference': [],
            'prompt_id': 44067482,
        }
    )

    assert sample.subset_key == 'writing'
    assert sample.input[0].text == 'Draft a message.'
    assert sample.metadata['turns'] == ['Draft a message.', 'Rewrite it as a poem.']


def test_two_turn_inference_preserves_generated_context_and_official_temperature() -> None:
    adapter = make_adapter()
    sample = adapter.record_to_sample(
        {
            'category': 'writing',
            'prompt': ['Write a greeting.', 'Make it rhyme.'],
            'reference': [],
            'prompt_id': 2,
        }
    )
    model = TwoTurnModel(['Hello.', 'Hello, yellow.'])

    result = adapter._on_inference(model, sample)

    assert result.output.completion == 'Hello, yellow.'
    assert [message.role for message in model.calls[1][0]] == ['user', 'assistant', 'user']
    assert model.calls[1][0][1].text == 'Hello.'
    assert model.calls[0][1].temperature == 0.7
    assert model.calls[0][1].max_tokens == 1024
    assert [message.text for message in result.messages] == ['Write a greeting.', 'Hello.', 'Make it rhyme.', 'Hello, yellow.']


def test_judge_scores_both_turns_and_uses_general_prompt() -> None:
    adapter = make_adapter()
    judge = ScriptedJudge([
        json.dumps({'explanation': 'Good first answer.', 'score': 8}),
        json.dumps({'explanation': 'Good follow-up.', 'score': 6}),
    ])
    adapter.llm_judge = judge

    score = adapter.calculate_metrics(make_state(adapter)).score

    assert score.status is ScoreStatus.SUCCESS
    assert score.value == {'judge_score': 7.0, 'first_turn_judge_score': 8.0, 'second_turn_judge_score': 6.0}
    assert 'First question' in judge.calls[0][1].text
    assert 'First reference' not in judge.calls[0][1].text
    assert 'Second answer' in judge.calls[1][1].text


def test_reference_guided_judge_uses_dataset_references() -> None:
    adapter = make_adapter()
    judge = ScriptedJudge([
        json.dumps({'explanation': 'Correct.', 'score': 10}),
        json.dumps({'explanation': 'Correct follow-up.', 'score': 9}),
    ])
    adapter.llm_judge = judge

    score = adapter.calculate_metrics(make_state(adapter, category='math')).score

    assert score.value['judge_score'] == 9.5
    assert 'First reference' in judge.calls[0][1].text
    assert 'First reference' in judge.calls[1][1].text
    assert 'Second reference' in judge.calls[1][1].text
