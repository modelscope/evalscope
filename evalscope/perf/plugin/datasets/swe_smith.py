"""SWE-smith multi-turn dataset plugin for evalscope perf benchmarking.

This plugin supports two data-source modes:

1. **Pre-built JSON** (``args.dataset_path`` is set):
   Load a pre-constructed ``agentic_dataset.json`` produced by
   ``data/swe_smith/build_dataset.py``.  Each conversation is a list of turns,
   where each turn stores only the DELTA messages for that turn.

2. **Live construction** (``args.dataset_path`` is ``None``):
   Pull the raw SWE-smith-trajectories dataset from ModelScope and construct
   conversations on-the-fly using the core logic ported from
   ``build_dataset.py``.  Requires ``args.tokenizer_path`` for accurate token
   counting; falls back to ``chars_per_token`` estimation otherwise.

Dataset format loaded from JSON
--------------------------------
The ``agentic_dataset.json`` structure::

    {
        "metadata": {...},
        "conversations": [
            [  # conversation 0
                {"messages": [...], "prompt_tokens": 65432, "output_tokens": 300},
                {"messages": [...], "prompt_tokens": 66100, "output_tokens": 300},
            ],
            ...
        ]
    }

Each turn's ``messages`` list contains only the **delta** messages to append
for that turn (not the full history).  The benchmark runner accumulates the
full context by prepending previous turns + model responses.

Plugin output format
--------------------
``build_messages()`` yields each conversation as a flat ``List[Dict]``
representing the full message sequence (all turns joined), compatible with
the multi-turn benchmark runner::

    [
        {'role': 'user', 'content': '...'},     # turn 1 message(s)
        {'role': 'user', 'content': '...'},     # turn 2 delta
        ...
    ]

The runner re-sends the growing context; the assistant responses produced
by the model fill the gaps between delta turns at runtime.
"""

import json
from typing import Any, Dict, Iterator, List, Optional

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase, Message, Messages
from evalscope.perf.plugin.datasets.utils import tokenize_chat_messages
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.logger import get_logger

logger = get_logger()

# Default ModelScope dataset used for live construction
_DEFAULT_DATASET_NAME = 'SWE-bench/SWE-smith-trajectories'
_DEFAULT_SPLIT = 'tool'

# ---------------------------------------------------------------------------
# Core conversation-building helpers (ported from build_dataset.py)
# ---------------------------------------------------------------------------


def _extract_messages(raw_messages: str) -> List[Dict[str, str]]:
    """Parse the messages JSON string and normalize to [{role, content}, ...]."""
    parsed = json.loads(raw_messages)
    messages = []
    for msg in parsed:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = '\n'.join(text_parts)
        if not content or not role:
            continue
        # Normalize role: tool -> user for chat template compatibility
        if role == 'tool':
            role = 'user'
        if role not in ('system', 'user', 'assistant'):
            continue
        messages.append({'role': role, 'content': content})
    return messages


def _count_tokens_for_messages(messages: List[Dict[str, str]], tokenizer) -> int:
    """Count tokens for a list of chat messages using apply_chat_template."""
    if not messages:
        return 0
    return len(tokenize_chat_messages(tokenizer, messages))


def _truncate_message_content(
    message: Dict[str, str],
    tokens_needed: int,
    tokenizer,
) -> Optional[Dict[str, str]]:
    """Truncate a message's content to approximately tokens_needed tokens."""
    if tokens_needed <= 0:
        return None
    content_tokens = tokenizer.encode(message['content'], add_special_tokens=False)
    truncated_content = tokenizer.decode(content_tokens[:tokens_needed], skip_special_tokens=True)
    return {'role': message['role'], 'content': truncated_content}


def _accumulate_messages(
    messages,
    msg_idx,
    msg_char_offset,
    current_messages,
    current_tokens,
    target_tokens,
    tokenizer,
):
    """Append messages from the trajectory until reaching target_tokens."""
    delta = []

    while msg_idx < len(messages):
        msg = messages[msg_idx]
        if msg_char_offset > 0:
            msg = {'role': msg['role'], 'content': msg['content'][msg_char_offset:]}
            if not msg['content']:
                msg_idx += 1
                msg_char_offset = 0
                continue

        trial_messages = current_messages + delta + [msg]
        trial_tokens = _count_tokens_for_messages(trial_messages, tokenizer)

        if trial_tokens < target_tokens:
            delta.append(msg)
            current_tokens = trial_tokens
            msg_idx += 1
            msg_char_offset = 0
        else:
            if msg['role'] == 'assistant':
                msg_idx += 1
                msg_char_offset = 0
                continue
            if trial_tokens == target_tokens:
                delta.append(msg)
                current_tokens = trial_tokens
                msg_idx += 1
                msg_char_offset = 0
            else:
                tokens_needed = target_tokens - current_tokens
                truncated = _truncate_message_content(msg, tokens_needed, tokenizer)
                if truncated:
                    delta.append(truncated)
                    consumed_chars = len(truncated['content'])
                    original_content = messages[msg_idx]['content']
                    msg_char_offset = (msg_char_offset or 0) + consumed_chars
                    if msg_char_offset >= len(original_content):
                        msg_idx += 1
                        msg_char_offset = 0
                else:
                    msg_idx += 1
                    msg_char_offset = 0
            break

    while delta and delta[-1]['role'] == 'assistant':
        delta.pop()

    if delta:
        current_tokens = _count_tokens_for_messages(current_messages + delta, tokenizer)

    return delta, current_tokens, msg_idx, msg_char_offset


def _make_fake_response(tokenizer, num_tokens: int) -> str:
    """Generate a fake response by decoding num_tokens token IDs."""
    return tokenizer.decode(list(range(1000, 1000 + num_tokens)), skip_special_tokens=True)


def _build_conversation(
    messages: List[Dict[str, str]],
    tokenizer,
    first_turn_length: int,
    subsequent_turn_length: int,
    max_context_length: int,
    output_length: int,
) -> Optional[List[Dict]]:
    """Build a multi-turn conversation from trajectory messages.

    Returns a list of turn dicts (``{"messages": [...], "prompt_tokens": int,
    "output_tokens": int}``), or ``None`` if the trajectory is too short.
    """
    turns = []
    accumulated: List[Dict] = []
    msg_idx = 0
    msg_char_offset = 0
    current_tokens = 0

    # Build first turn
    delta, current_tokens, msg_idx, msg_char_offset = _accumulate_messages(
        messages, msg_idx, msg_char_offset, accumulated, current_tokens, first_turn_length, tokenizer
    )

    if current_tokens < first_turn_length * 0.8:
        return None

    accumulated.extend(delta)
    turns.append({'messages': delta, 'prompt_tokens': current_tokens, 'output_tokens': output_length})

    fake_response_text = _make_fake_response(tokenizer, output_length)

    # Build subsequent turns
    while msg_idx < len(messages) and current_tokens + output_length < max_context_length:
        fake_response = {'role': 'assistant', 'content': fake_response_text}
        accumulated.append(fake_response)
        current_tokens = _count_tokens_for_messages(accumulated, tokenizer)

        if current_tokens + output_length >= max_context_length:
            break

        if (msg_char_offset == 0 and msg_idx < len(messages) and messages[msg_idx]['role'] == 'assistant'):
            msg_idx += 1

        if msg_idx >= len(messages):
            break

        target = current_tokens + subsequent_turn_length
        if target + output_length > max_context_length:
            break

        delta, current_tokens, msg_idx, msg_char_offset = _accumulate_messages(
            messages, msg_idx, msg_char_offset, accumulated, current_tokens, target, tokenizer
        )

        if not delta or abs(current_tokens - target) > 5:
            break

        accumulated.extend(delta)
        turns.append({'messages': delta, 'prompt_tokens': current_tokens, 'output_tokens': output_length})

    remaining = max_context_length - output_length - first_turn_length
    step = subsequent_turn_length + output_length
    expected_turns = 1 + remaining // step

    if len(turns) in (expected_turns, expected_turns - 1):
        return turns
    return None


# ---------------------------------------------------------------------------
# Dataset plugin
# ---------------------------------------------------------------------------


@register_dataset('swe_smith')
class SweSmithDatasetPlugin(DatasetPluginBase):
    """Multi-turn dataset plugin backed by SWE-smith-trajectories.

    Two modes:

    * **Pre-built** (``--dataset-path``): Load ``agentic_dataset.json``
      directly.  Each conversation is a list of turn dicts with ``messages``
      (delta) and ``prompt_tokens`` / ``output_tokens`` metadata.

    * **Live construction** (no ``--dataset-path``): Stream trajectories from
      ModelScope, apply the ``build_dataset.py`` logic, and yield conversations
      on-the-fly.  Token-length parameters are taken from ``multi_turn_args``
      (supports range sampling via seed) or sensible defaults.

    Parameters from ``multi_turn_args`` that affect live construction:

    * ``first_turn_length``   – target tokens for turn 1 (IntOrRange)
    * ``subsequent_turn_length`` – token growth per subsequent turn (IntOrRange)
    * ``max_context_length``  – max total context tokens (IntOrRange)
    * ``chars_per_token``     – pre-filter estimate (no tokenizer fallback)
    * ``offset``              – shuffle offset to avoid cache hits
    * ``max_turns``           – max user turns to include per conversation
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_messages(self) -> Iterator[List[Dict]]:
        """Yield conversations as flat lists of OpenAI-style message dicts."""
        if self.query_parameters.dataset_path:
            yield from self._load_from_json()
        else:
            yield from self._load_live()

    # ------------------------------------------------------------------
    # Pre-built JSON mode
    # ------------------------------------------------------------------

    def _load_from_json(self) -> Iterator[List[Messages]]:
        """Load pre-built ``agentic_dataset.json`` and yield conversations."""
        dataset_path = self.query_parameters.dataset_path
        logger.info(f'Loading pre-built dataset from {dataset_path}')

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = data.get('conversations', [])
        if not conversations:
            logger.warning(f'No conversations found in {dataset_path}')
            return

        # Apply offset: shuffle conversations deterministically before slicing
        mt_args = self.query_parameters.multi_turn_args
        offset = self.query_parameters.dataset_offset
        if offset > 0:
            conversations = conversations[offset:] + conversations[:offset]

        # Determine max_turns cap
        max_turns = mt_args.max_turns if mt_args else None

        logger.info(f'Loaded {len(conversations)} conversations '
                    f'(offset={offset}, max_turns={max_turns})')

        for conversation in conversations:
            # Each turn's "messages" list is already the delta (non-assistant).
            turns: List[Messages] = [turn.get('messages', []) for turn in conversation]

            # Apply max_turns truncation at the dataset layer
            if max_turns is not None:
                turns = turns[:max_turns]

            if turns:
                yield turns

    # ------------------------------------------------------------------
    # Live construction mode
    # ------------------------------------------------------------------

    def _load_live(self) -> Iterator[List[Messages]]:
        """Pull SWE-smith-trajectories from ModelScope and construct conversations."""
        if self.tokenizer is None:
            raise ValueError(
                'Live construction mode requires --tokenizer-path for accurate '
                'token counting.  Please specify --tokenizer-path or provide a pre-built '
                '--dataset-path.'
            )

        mt_args = self.query_parameters.multi_turn_args

        # Read static / non-sampleable params directly
        chars_per_token = mt_args.chars_per_token if mt_args else 3.0
        offset = self.query_parameters.dataset_offset
        min_turns = mt_args.min_turns if mt_args else 1
        max_turns = mt_args.max_turns if mt_args else 5
        output_length = self.query_parameters.max_tokens or 300

        # For pre-filtering, use the upper bound of max_context_length so that
        # we don't discard candidates that might be viable under a smaller sample.
        if mt_args is not None:
            max_ctx_upper = (
                mt_args.max_context_length
                if isinstance(mt_args.max_context_length, int) else mt_args.max_context_length[1]
            )
        else:
            max_ctx_upper = 75000

        logger.info(
            f'Live construction: offset={offset}, min_turns={min_turns}, max_turns={max_turns}, '
            f'output_length={output_length}, max_ctx_upper={max_ctx_upper}'
        )

        from modelscope import MsDataset
        dataset = MsDataset.load(_DEFAULT_DATASET_NAME, split=_DEFAULT_SPLIT)

        target_count = max(1, self.query_parameters.number)
        # Scan at most scan_multiplier × target_count rows; _build_conversation has a
        # ~50 % pass rate so 20× gives a comfortable safety margin.
        scan_multiplier = 20
        max_scan = target_count * scan_multiplier

        min_chars = int(max_ctx_upper * chars_per_token)
        candidates = []
        skipped_short = 0
        skipped_parse = 0

        for row in dataset:
            raw_messages = row.get('messages', '') if isinstance(row, dict) else row['messages']
            if len(raw_messages) < min_chars:
                skipped_short += 1
                continue
            try:
                messages = _extract_messages(raw_messages)
            except (json.JSONDecodeError, KeyError, TypeError):
                skipped_parse += 1
                continue
            total_chars = sum(len(m['content']) for m in messages)
            if total_chars < min_chars:
                skipped_short += 1
                continue
            candidates.append(messages)
            if len(candidates) >= max_scan:
                logger.info(f'Reached scan limit ({max_scan}), stopping dataset scan early.')
                break

        logger.info(
            f'Pre-filter: {len(candidates)} candidates '
            f'({skipped_short} too short, {skipped_parse} parse error)'
        )

        import random
        random.shuffle(candidates)

        # Apply offset
        if offset > 0:
            candidates = candidates[offset:] + candidates[:offset]

        # Build conversations one-by-one; sample IntOrRange params PER conversation
        # so that different conversations can have different token-length targets.
        built = 0
        for messages in candidates:
            if built >= target_count:
                break
            sampled = self.get_sampled_multi_turn_params()
            first_turn_length = sampled.get('first_turn_length', 65000)
            subsequent_turn_length = sampled.get('subsequent_turn_length', 500)
            max_context_length = sampled.get('max_context_length', 75000)

            conversation = _build_conversation(
                messages,
                self.tokenizer,
                first_turn_length,
                subsequent_turn_length,
                max_context_length,
                output_length,
            )
            if conversation is None:
                continue

            # Each turn's delta is the Messages for that turn.
            turns: List[Messages] = [turn.get('messages', []) for turn in conversation]

            # Apply max_turns truncation at the dataset layer
            if max_turns is not None:
                turns = turns[:max_turns]

            if len(turns) >= min_turns:
                built += 1
                yield turns
