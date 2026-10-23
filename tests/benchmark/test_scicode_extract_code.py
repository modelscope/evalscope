# Copyright (c) Alibaba, Inc. and its affiliates.
"""SciCode runs the code inside a response's code block, not the text the model writes around it."""
import ast

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.model import ModelOutput
from evalscope.benchmarks.scicode.util import extract_code, get_generated_code


def test_text_around_the_code_block_is_dropped() -> None:
    answers = [
        'Here is the implementation:\n```python\ndef f(x):\n    return x + 1\n```\nIt adds one.',
        '```python\ndef g(x):\n    return f(x) * 2\n```',
    ]
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='', metadata={'sub_steps': [{}, {}]}, id=0),
        output=ModelOutput.from_content('mock', ''),
        completed=True,
    )
    state.messages = []
    for answer in answers:
        state.messages += [ChatMessageUser(content='step'), ChatMessageAssistant(content=answer)]

    codes = get_generated_code(state)

    assert codes == ['def f(x):\n    return x + 1', 'def g(x):\n    return f(x) * 2']
    ast.parse('\n'.join(codes))  # the script of the last subproblem includes every earlier answer


def test_response_without_fences_is_used_whole() -> None:
    assert extract_code('def f(x):\n    return x\n') == 'def f(x):\n    return x'
