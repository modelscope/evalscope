from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageAssistant, ContentText
from evalscope.benchmarks.scicode.util import get_generated_code


def test_get_generated_code_accepts_text_content_blocks():
    state = TaskState(
        model="test",
        sample=Sample(input="77", metadata={"sub_steps": [{}]}),
        messages=[
            ChatMessageAssistant(
                content=[
                    ContentText(text="```python\ndef answer():\n    return 1\n```"),
                ],
            ),
        ],
    )

    assert get_generated_code(state) == ["def answer():\n    return 1"]
