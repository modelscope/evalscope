"""Trie agentic-workload trace-replay dataset plugins.

Replays production agentic traces published in [applied-compute/trie](https://github.com/applied-compute/trie)
(Apache-2.0).  Each trace is a multi-turn conversation described by token-length
sequences and simulated tool-call wait times; the runner synthesizes prompts to
the recorded lengths, sleeps ``tool_call_latency`` seconds between turns, and
caps each assistant response with the recorded ``assistant_response_length``.

Three workloads are re-hosted on ModelScope as ``evalscope/trie-workloads``:

* ``agentic_coding_8k.jsonl`` - coding-agent traces, ~8k context
* ``code_qa_8k.jsonl`` - code Q&A traces, ~8k context
* ``office_work_8k.jsonl`` - office-work agent traces, ~8k context

Trace schema (one JSON object per line):

* ``input_prompt_length`` (int): initial user prompt length in tokens
* ``num_turns`` (int): number of assistant + tool turns excluding the final reply
* ``assistant_response_length`` (List[int], len=num_turns): per-turn output cap
* ``tool_call_output_length`` (List[int], len=num_turns): tokens injected as tool output
* ``tool_call_latency`` (List[float], len=num_turns): seconds to sleep before next turn
* ``final_assistant_response_length`` (int): output cap of the final assistant reply

Each trace yields ``num_turns + 1`` Turn objects.  Requests must set
``ignore_eos`` so the recorded length sequences are honored exactly; pass it via
``--extra-args '{"ignore_eos": true}'``.

Note on numerical alignment with the upstream ``trie`` tool: ``trie`` sends
``/v1/completions`` with raw prompt strings, while evalscope multi-turn sends
``/v1/chat/completions``.  The server-side chat template adds ~10-20 tokens of
overhead per turn, so prompt-token counts and cache-hit-rate numbers will
differ from ``trie`` by a few percent.  This is inherent to the chat-template
path and is not a bug.
"""

import json
import numpy as np
import os
from typing import Any, Dict, Iterator

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import Conversation, Turn
from evalscope.perf.plugin.datasets.random_dataset import RandomDatasetPlugin
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.logger import get_logger

logger = get_logger()

_HUB_REPO = 'evalscope/trie-workloads'


class TrieReplayBase(RandomDatasetPlugin):
    """Base class for trie trace-replay plugins.

    Subclasses set ``FILE_NAME`` to one of the three jsonl files re-hosted under
    ``evalscope/trie-workloads`` on ModelScope.

    Inherits from :class:`RandomDatasetPlugin` to reuse:

    * ``self.allowed_tokens`` (precomputed once in ``__init__``; excludes special
      tokens **and** byte-fallback tokens that would otherwise inflate
      multi-byte content by 3-5x on the server side);
    * ``generate_token_sequence`` (decode/re-encode repair loop that ensures the
      server tokenizer sees exactly ``target_len`` tokens).
    """

    FILE_NAME: str = ''  # subclasses must override

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        if not query_parameters.multi_turn:
            logger.warning(
                'trie trace-replay is a multi-turn dataset; pass --multi-turn so each trace '
                'is replayed as one multi-turn conversation against the server.'
            )

    def _resolve_dataset_path(self) -> str:
        if self.query_parameters.dataset_path:
            return self.query_parameters.dataset_path
        from modelscope import dataset_snapshot_download
        local_path = dataset_snapshot_download(_HUB_REPO, allow_patterns=[self.FILE_NAME])
        path = os.path.join(local_path, self.FILE_NAME)
        self.query_parameters.dataset_path = path
        return path

    def _synth_prompt(self, target_len: int) -> str:
        """Return a synthetic prompt that re-encodes to ``target_len`` tokens.

        Always synthesizes at least one token so the resulting user message is
        never empty (some chat templates reject empty ``content``).
        """
        target_len = max(1, int(target_len))
        offset = int(np.random.randint(0, len(self.allowed_tokens)))
        prompt, _, _ = self.generate_token_sequence(input_len=target_len, offset=offset, index=0)
        return prompt

    def _trace_to_conversation(self, trace: Dict[str, Any]) -> Conversation:
        """Convert one jsonl trace row to a ``Conversation`` (``List[Turn]``).

        Mapping from trace schema to turns (total = ``num_turns + 1`` turns):

        * Turn 0: initial user prompt of ``input_prompt_length`` tokens;
          ``max_tokens = assistant_response_length[0]``; no inter-turn sleep.
        * Turn i (1 ≤ i ≤ num_turns - 1): tool-output user message of
          ``tool_call_output_length[i-1]`` tokens; sleep ``tool_call_latency[i-1]``
          before sending; ``max_tokens = assistant_response_length[i]``.
        * Turn num_turns (final): tool-output user message of
          ``tool_call_output_length[num_turns-1]`` tokens; sleep
          ``tool_call_latency[num_turns-1]`` before sending;
          ``max_tokens = final_assistant_response_length``;
          ``is_final = True``.
        """
        input_prompt_length = int(trace['input_prompt_length'])
        num_turns = int(trace['num_turns'])
        assistant_response_length = trace['assistant_response_length']
        tool_call_output_length = trace['tool_call_output_length']
        tool_call_latency = trace['tool_call_latency']
        final_response_length = int(trace['final_assistant_response_length'])

        if not (
            len(assistant_response_length) == num_turns and len(tool_call_output_length) == num_turns
            and len(tool_call_latency) == num_turns
        ):
            raise ValueError(
                f'trace length mismatch: num_turns={num_turns}, '
                f'len(assistant_response_length)={len(assistant_response_length)}, '
                f'len(tool_call_output_length)={len(tool_call_output_length)}, '
                f'len(tool_call_latency)={len(tool_call_latency)}'
            )

        turns: Conversation = []
        # Turn 0: initial user prompt
        turns.append(
            Turn(
                messages=[{
                    'role': 'user',
                    'content': self._synth_prompt(input_prompt_length)
                }],
                max_tokens=int(assistant_response_length[0]) if num_turns > 0 else final_response_length,
                tool_call_latency=None,
                is_final=(num_turns == 0),
            )
        )

        # Turns 1..num_turns: tool output + assistant response cap
        for i in range(num_turns):
            is_last = (i == num_turns - 1)
            next_max_tokens = (final_response_length if is_last else int(assistant_response_length[i + 1]))
            turns.append(
                Turn(
                    messages=[{
                        'role': 'user',
                        'content': self._synth_prompt(int(tool_call_output_length[i])),
                    }],
                    max_tokens=next_max_tokens,
                    tool_call_latency=float(tool_call_latency[i]),
                    is_final=is_last,
                )
            )

        return turns

    def build_messages(self) -> Iterator[Conversation]:
        """Yield each trace as a ``Conversation`` of pre-synthesized turns."""
        path = self._resolve_dataset_path()
        for line in self.dataset_line_by_line(path):
            line = line.strip()
            if not line:
                continue
            try:
                trace = json.loads(line)
                yield self._trace_to_conversation(trace)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(f'Skipping malformed trace: {e}')
                continue


@register_dataset('trie_agentic_coding')
class TrieAgenticCodingPlugin(TrieReplayBase):
    """Coding-agent traces (~8k context). Source: applied-compute/trie."""
    FILE_NAME = 'agentic_coding_8k.jsonl'


@register_dataset('trie_code_qa')
class TrieCodeQaPlugin(TrieReplayBase):
    """Code Q&A traces (~8k context). Source: applied-compute/trie."""
    FILE_NAME = 'code_qa_8k.jsonl'


@register_dataset('trie_office_work')
class TrieOfficeWorkPlugin(TrieReplayBase):
    """Office-work agent traces (~8k context). Source: applied-compute/trie."""
    FILE_NAME = 'office_work_8k.jsonl'
