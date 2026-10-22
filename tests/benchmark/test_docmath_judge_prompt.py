"""DocMath shows the judge the question it asks about, as FRAMES does."""
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeContext
from evalscope.api.judge.types import Placement
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def test_judge_prompt_contains_the_question() -> None:
    adapter = get_benchmark('docmath', TaskConfig(model='mock', datasets=['docmath']))
    question = 'What was the year-over-year revenue growth in percent?'
    sample = adapter.record_to_sample({'question': question, 'paragraphs': ['Revenue: 100, then 112.'], 'ground_truth': 12.0})
    reply = 'Therefore, the answer is 12.'
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content(model='mock', content=reply))
    context = JudgeContext(task_state=state, original_prediction=reply, filtered_prediction='12', reference=sample.target)
    definition = adapter.judge_definition(context)

    request = definition.request(definition.cases[0], Placement.ORIGINAL, [], context)

    assert f'Problem: {question}\n' in request.messages[-1].text
