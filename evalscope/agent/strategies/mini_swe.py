"""mini-swe-agent strategy (textual_block mode).

A minimal strategy where the model outputs ``THOUGHT`` lines followed by a
`````bash`` code block containing one shell command.  The strategy parses
the command, synthesises a :class:`ToolCall` for the ``bash`` tool, and
the loop executes it via the :class:`ToolExecutor`.

Task completion is signalled by the sentinel string
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` appearing in the bash output.
This is the textual equivalent of calling the ``submit`` tool in FC mode.
"""

import re
import uuid
from typing import List, Optional

from evalscope.api.agent import AgentContext, AgentLoopResult, AgentStrategy, ParsedAction, ToolSchemaMode
from evalscope.api.messages import ChatMessage, ChatMessageUser
from evalscope.api.model import ModelOutput
from evalscope.api.registry import register_strategy
from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction, ToolInfo

# Sentinel that the model prints to signal task completion.
SUBMIT_SENTINEL = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'

# Regex for extracting ```bash ... ``` blocks from model output.
_BASH_BLOCK_RE = re.compile(r'```bash\n(.*?)\n```', re.DOTALL)

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

MINI_SWE_SYSTEM_PROMPT = """\
You are an expert software engineer.  Your task is to solve the given problem.

You have access to a bash shell.  You can use any bash command to explore, compute, and verify.

You MUST respond in this EXACT format:

THOUGHT: <your reasoning about what to do next>

```bash
<exactly one bash command>
```

Rules:
1. ALWAYS start with THOUGHT:
2. ALWAYS provide exactly ONE bash command in a ```bash block
3. Wait for the command result before your next response
4. Use 'cat' to read files, 'grep' to search, 'sed' to edit
5. Use 'python3 -c "..."' for complex calculations
6. When done, output the following marker on its own line in your bash output:

   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

   followed by the final answer.

Recommended workflow:
1. First, understand the problem
2. Develop a solution strategy
3. Implement and verify
4. Submit with the COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT marker
"""


@register_strategy('mini_swe')
class MiniSweStrategy:
    """Minimal SWE-agent strategy using text-parsed bash blocks."""

    name: str = 'mini_swe'

    def __init__(self, *, system_prompt: Optional[str] = None, **_: object) -> None:
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # AgentStrategy implementation
    # ------------------------------------------------------------------

    def build_system_prompt(self, ctx: AgentContext) -> Optional[str]:
        return self._system_prompt or MINI_SWE_SYSTEM_PROMPT

    def prepare_messages(self, ctx: AgentContext) -> List[ChatMessage]:
        return ctx.messages

    def parse_output(self, output: ModelOutput, ctx: AgentContext) -> ParsedAction:
        content = output.message.text or ''

        # Extract ```bash ... ``` blocks.
        bash_blocks = _BASH_BLOCK_RE.findall(content)

        if not bash_blocks:
            # No bash block found → treat the response as a final answer
            # or a format error.  If the content is non-empty we treat it
            # as a final answer so the loop can terminate gracefully.
            return ParsedAction(final_answer=content, raw_text=content)

        command = bash_blocks[0].strip()
        call = ToolCall(
            id=f'swe_{uuid.uuid4().hex[:8]}',
            function=ToolFunction(name='bash', arguments={'command': command}),
        )
        return ParsedAction(tool_calls=[call], raw_text=content)

    def is_done(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        # Explicit final answer from parse_output (no bash block found).
        if parsed.final_answer is not None:
            return True

        # Post-execution sentinel check: scan recent observations for
        # COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.
        for msg in reversed(ctx.messages[-3:]):
            if msg.role in ('tool', 'user'):
                if SUBMIT_SENTINEL in str(msg.content):
                    return True
        return False

    def should_nudge(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        # mini_swe never nudges; if no bash block found, parse_output already
        # sets final_answer and the loop terminates via is_done.
        return False

    def tool_schema_mode(self) -> ToolSchemaMode:
        return 'textual_block'

    def tools(self, ctx: AgentContext) -> List[ToolInfo]:
        # textual_block mode does not pass tool schemas to model.generate.
        return []

    def format_observation(
        self,
        call: ToolCall,
        observation: str,
        error: Optional[ToolCallError],
    ) -> ChatMessage:
        # textual_block models expect observations as user messages,
        # not tool messages (they don't use function-calling).
        if error:
            content = f'ERROR: {error.message}'
        else:
            content = observation
        return ChatMessageUser(content=content)

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Extract the final answer from the message history.

        Scans observations for the COMPLETE_TASK sentinel (equivalent to
        the ``submit`` tool in FC mode).  Falls back to the last
        assistant message text.
        """
        # Check for sentinel in observations.
        for msg in reversed(result.messages):
            if msg.role in ('tool', 'user'):
                content = str(msg.content)
                match = re.search(
                    rf'{re.escape(SUBMIT_SENTINEL)}\s*(.*)',
                    content,
                    re.DOTALL,
                )
                if match:
                    answer = match.group(1).strip()
                    if answer:
                        return answer

        # Fallback: last assistant text.
        for msg in reversed(result.messages):
            if msg.role == 'assistant':
                text = msg.text
                if text:
                    return text

        return ''


__all__ = ['MiniSweStrategy', 'SUBMIT_SENTINEL']
