"""Regression coverage for literal values in judge prompt templates."""

from typing import List, Optional

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import JudgeScoreType, ScoreStatus
from evalscope.metrics.judge.llm_judge import DEFAULT_NUMERIC_SCORE_TEMPLATE, DEFAULT_PROMPT_TEMPLATE, LLMJudge


def make_judge(template: str) -> LLMJudge:
    """Create a prompt renderer without initializing a model or network client."""
    judge = object.__new__(LLMJudge)
    judge.prompt_template = template
    return judge


@pytest.mark.parametrize('field', ['question', 'pred', 'gold'])
@pytest.mark.parametrize('placeholder', ['{question}', '{pred}', '{gold}'])
def test_inserted_placeholders_remain_literal(field: str, placeholder: str) -> None:
    values = {'question': 'QUESTION', 'pred': 'PREDICTION', 'gold': 'REFERENCE'}
    values[field] = placeholder
    judge = make_judge('Q={question}\nP={pred}\nG={gold}')

    assert judge.build_prompt(**values) == f"Q={values['question']}\nP={values['pred']}\nG={values['gold']}"


@pytest.mark.parametrize('question', [None, '', 'Question?'])
@pytest.mark.parametrize('template', [DEFAULT_PROMPT_TEMPLATE, DEFAULT_NUMERIC_SCORE_TEMPLATE], ids=['pattern', 'numeric'])
def test_default_templates_preserve_answer_content(template: str, question: Optional[str]) -> None:
    prediction = '{gold} {pred} {question}'
    reference = 'REFERENCE'
    prompt = make_judge(template).build_prompt(prediction, reference, question)

    assert f"[Question]\n{'Not provided' if question is None else question}\n" in prompt
    section = '[Predicted Answer]' if template == DEFAULT_PROMPT_TEMPLATE else '[Response]'
    assert f'{section}\n{prediction}' in prompt
    if template == DEFAULT_PROMPT_TEMPLATE:
        assert f'[Reference Answer]\n{reference}' in prompt


def test_custom_template_preserves_json_unknown_fields_and_backslashes() -> None:
    template = '{"verdict": "A", "metadata": {}}\n{unknown}\n{pred}\n{pred}\n{gold}'
    prediction = r'\1\g<gold> {gold} {"answer": {}}'

    assert make_judge(template).build_prompt(prediction, 'REFERENCE') == (
        '{"verdict": "A", "metadata": {}}\n{unknown}\n'
        f'{prediction}\n{prediction}\nREFERENCE'
    )


class CapturingJudge:
    """Capture the real scoring request and return a fixed offline verdict."""

    score_type = JudgeScoreType.PATTERN
    score_mapping = {'A': 1.0, 'B': 0.0}
    prompt_template = DEFAULT_PROMPT_TEMPLATE
    system_prompt = None
    judge_id = model_id = 'offline'
    build_prompt = LLMJudge.build_prompt

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def generate(self, messages: List[ChatMessage]) -> ModelOutput:
        """Record the outgoing prompt without contacting a model."""
        self.prompts.append(messages[-1].content)
        return ModelOutput.from_content('offline', '{"verdict":"B"}')


def test_scoring_chain_sends_the_original_prediction_to_judge() -> None:
    config = TaskConfig(
        model='m',
        eval_type='mock_llm',
        datasets=['general_qa'],
        judge={'strategy': 'llm', 'models': [{'model_id': 'offline'}]},
    )
    adapter = get_benchmark('general_qa', config)
    judge = CapturingJudge()
    adapter.llm_judge = judge
    sample = Sample(id=0, input='Question?', target='CANARY_REFERENCE_1697')
    state = TaskState(
        model='m', sample=sample, output=ModelOutput.from_content('m', '{gold}'), completed=True
    )

    score = adapter.calculate_metrics(state).score

    assert len(judge.prompts) == 1
    assert '[Predicted Answer]\n{gold}\n\n' in judge.prompts[0]
    assert '[Reference Answer]\nCANARY_REFERENCE_1697\n\n' in judge.prompts[0]
    assert score.prediction == '{gold}'
    assert score.status is ScoreStatus.SUCCESS
    assert score.value == {'acc': 0.0}
