"""Submit tool: unified answer-submission / loop-termination mechanism.

When the model calls ``submit(answer="42")`` via function-calling, the
strategy intercepts it in ``parse_output`` and returns
``ParsedAction(final_answer="42")`` so the loop terminates without
executing the tool.  The handler registered here is a safety net that
is only reached if a strategy fails to intercept.
"""

from typing import Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_agent_tool
from evalscope.api.tool import ToolCall, ToolInfo
from evalscope.api.tool.tool_info import ToolParams
from evalscope.utils.json_schema import JSONSchema

SUBMIT_TOOL_INFO = ToolInfo(
    name='submit',
    description=(
        'Submit your final answer to complete the task. '
        'Call this when you have determined the answer and are confident. '
        'This will immediately terminate the agent loop.'
    ),
    parameters=ToolParams(
        properties={
            'answer': JSONSchema(
                type='string',
                description='The final answer to submit.',
            ),
        },
        required=['answer'],
    ),
)


@register_agent_tool('submit', info=SUBMIT_TOOL_INFO)
async def run_submit(call: ToolCall, env: Optional[AgentEnvironment]) -> str:
    """Submit the final answer.  Normally intercepted by the strategy before
    execution; this handler exists as a safety fallback."""
    answer = call.function.arguments.get('answer', '')
    return f'Task submitted. Answer: {answer}'


__all__ = ['SUBMIT_TOOL_INFO', 'run_submit']
