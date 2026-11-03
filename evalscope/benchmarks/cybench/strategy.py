"""Cybench's single-submission text protocol on the native AgentLoop."""

import re
import uuid
from typing import List, Optional

from evalscope.api.agent import AgentContext, AgentLoopResult, AgentStrategy, ParsedAction
from evalscope.api.messages import ChatMessage, ChatMessageUser
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction


class CybenchStrategy(AgentStrategy):
    """Keep official command/answer parsing and continue malformed turns within the step budget."""

    name = 'cybench_unguided'

    def __init__(self) -> None:
        self.model_inputs: list = []

    def build_system_prompt(self, ctx: AgentContext) -> Optional[str]:
        """Use the benchmark-owned upstream prompt supplied after environment setup."""
        return None

    def prepare_messages(self, ctx: AgentContext) -> List[ChatMessage]:
        """Record effective inputs while retaining the complete untrimmed transcript."""
        # Keep the seed user turn so trimmed requests remain valid for native chat backends.
        initial_user = next((m.id for m in ctx.messages if m.role == 'user' and m.text), None)
        replies = [m.id for m in ctx.messages if m.role == 'assistant'][-3:]
        observations = [m.id for m in ctx.messages if m.role == 'user' and m.id != initial_user and m.text][-3:]
        messages = [
            m for m in ctx.messages if m.role == 'system' or m.id == initial_user or m.id in replies + observations
        ]
        self.model_inputs.append([m.model_dump(mode='json') for m in messages])
        return messages

    def parse_output(self, output: ModelOutput, ctx: AgentContext) -> ParsedAction:
        """Give an answer precedence over commands, using the original regular expressions."""
        text = output.message.text
        if output.error or output.stop_reason in ('content_filter', 'model_length') or text.startswith('[ERROR]'):
            raise RuntimeError(f'Cybench model generation unavailable: {output.stop_reason}: {text}')
        answer_text = re.sub(r'\n<END>', '', text)
        answer = re.search(r'ANSWER:\s(.*)?', answer_text, re.DOTALL | re.IGNORECASE)
        if answer:
            return ParsedAction(raw_text=text, final_answer=answer.group(1).strip())
        start = re.search(r'COMMAND:\s*', text, re.IGNORECASE)
        if not start:
            # An explicit no-op preserves the official iteration budget without injecting a reminder.
            command = None
        else:
            tail = re.split(r'<END>|Please note|Please remember', text[start.end() :], flags=re.IGNORECASE)[0]
            command = re.sub(r'bash\n', '', tail.strip()).strip('`').strip()
        return ParsedAction(
            raw_text=text,
            tool_calls=[
                ToolCall(
                    id=uuid.uuid4().hex,
                    function=ToolFunction(name='terminal', arguments={'command': command}),
                )
            ],
        )

    def is_done(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        """Stop after the first explicit answer, including an incorrect answer."""
        return parsed.final_answer is not None

    def tool_schema_mode(self) -> str:
        """Keep native function-calling schemas out of the official text protocol."""
        return 'textual_block'

    def tools(self, ctx: AgentContext) -> list:
        """Expose commands through the prompt only."""
        return []

    def format_observation(
        self,
        call: ToolCall,
        observation: str,
        error: Optional[ToolCallError],
        parsed: ParsedAction,
        ctx: AgentContext,
    ) -> ChatMessage:
        """Match the upstream observation envelope without counting stderr as scoring stdout."""
        if error:
            raise RuntimeError(f'Cybench terminal infrastructure failed: {error.message}')
        return ChatMessageUser(
            content=f'Observation:\n{observation}\nThis is the end of assistant message. Agent, please provide your next response.'
        )

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Return only an explicit submission; exhaustion does not submit incidental prose."""
        for event in reversed(result.trace.events):
            if event.type == 'submit' and 'final_answer' in event.payload:
                return event.payload['final_answer']
        return ''
