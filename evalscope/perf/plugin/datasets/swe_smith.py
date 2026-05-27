"""SWE-smith multi-turn dataset plugin for evalscope perf benchmarking.

This plugin supports two data-source modes:

1. **Pre-built JSON** (``args.dataset_path`` is set):
   Load a pre-constructed ``agentic_dataset.json`` produced by
   ``examples/perf/build_swe_smith_dataset.py``.  Each conversation is a list
   of turns, where each turn stores only the DELTA messages for that turn.

2. **Live construction** (``args.dataset_path`` is ``None``):
   Pull the raw SWE-smith-trajectories dataset from ModelScope and construct
   conversations in parallel using ``multiprocessing.Pool``.  Requires
   ``args.tokenizer_path`` for accurate token counting.

   The entire dataset is scanned once and all trajectories that pass the
   character-length pre-filter are collected as candidates.  Candidates are
   then shuffled.  For each candidate ``num_turns`` is sampled uniformly from
   ``[min_turns, max_turns]`` (``--min-turns`` / ``--max-turns``).  All work
   items are dispatched to a ``multiprocessing.Pool``
   (size ``multi_turn_args.num_workers``); results are collected until
   ``number`` valid conversations have been built.
"""

import json
import multiprocessing
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, Iterator, List, Optional, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import Conversation, DatasetPluginBase, Message, Messages, Turn
from evalscope.perf.plugin.datasets.utils import tokenize_chat_messages
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.logger import get_logger

logger = get_logger()

# Default ModelScope dataset used for live construction
_DEFAULT_DATASET_NAME = 'SWE-bench/SWE-smith-trajectories'
_DEFAULT_SPLIT = 'tool'

# ---------------------------------------------------------------------------
# Core conversation-building helpers (aligned with build_swe_smith_dataset.py)
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


def _bare_encode(text: str, tokenizer) -> List[int]:
    """Encode text to token ids without special tokens."""
    return tokenizer.encode(text, add_special_tokens=False)


def _encode_len(text: str, tokenizer) -> int:
    """Fast bare-text token count (no chat template overhead)."""
    return len(_bare_encode(text, tokenizer))


def _truncate_message_content(
    message: Dict[str, str],
    tokens_needed: int,
    tokenizer,
) -> Dict[str, str]:
    """Truncate a message's content so it occupies at most tokens_needed bare tokens."""
    content_tokens = _bare_encode(message['content'], tokenizer)
    truncated_content = tokenizer.decode(content_tokens[:tokens_needed], skip_special_tokens=True)
    return {'role': message['role'], 'content': truncated_content}


def _collect_until(
    user_msgs: List[Dict[str, str]],
    start_idx: int,
    target_tokens: int,
    tokenizer,
) -> Tuple[List[Dict[str, str]], int]:
    """Collect user messages from start_idx until the prompt grows by ~target_tokens.

    Uses fast bare-text token counting (no chat template) to estimate the
    incremental size of each message.  A message that fits exactly within the
    remaining budget is collected whole.  The last message is truncated only
    when it would *exceed* the budget (``remaining > 0``); if the budget is
    already exhausted the message is left unconsumed (``idx`` not advanced).

    Args:
        user_msgs: Full list of user-only messages for this trajectory.
        start_idx: Index in user_msgs to start collecting from.
        target_tokens: Desired incremental token growth (bare-token estimate).
        tokenizer: Tokenizer used for token counting.

    Returns:
        (delta, new_idx) where delta is the list of collected messages and
        new_idx is the index of the next unconsumed message.
    """
    delta: List[Dict[str, str]] = []
    accumulated_bare = 0
    idx = start_idx

    while idx < len(user_msgs):
        msg = user_msgs[idx]
        msg_bare = _encode_len(msg['content'], tokenizer)

        if accumulated_bare + msg_bare <= target_tokens:
            # Message fits entirely within the remaining budget.
            delta.append(msg)
            accumulated_bare += msg_bare
            idx += 1
        else:
            remaining = target_tokens - accumulated_bare
            if remaining <= 0:
                # Budget already exhausted; do NOT consume this message.
                break
            truncated = _truncate_message_content(msg, remaining, tokenizer)
            delta.append(truncated)
            idx += 1
            break

    return delta, idx


def _build_one(args_tuple: Tuple) -> Optional[List[Dict]]:
    """Worker function for multiprocessing.Pool.

    Unpacks a tuple of (messages, tokenizer, first_turn_length,
    subsequent_turn_length, num_turns) and calls _build_conversation.
    """
    messages, tokenizer, first_turn_length, subsequent_turn_length, num_turns = args_tuple
    return _build_conversation(
        messages,
        tokenizer,
        first_turn_length,
        subsequent_turn_length,
        num_turns,
    )


def _build_conversation(
    messages: List[Dict[str, str]],
    tokenizer,
    first_turn_length: int,
    subsequent_turn_length: int,
    num_turns: int,
) -> Optional[List[Dict]]:
    """Build a multi-turn conversation from a trajectory.

    Only user messages from the trajectory are used.  Each turn stores the DELTA
    user messages for that turn (not the full history).  Assistant messages in the
    source trajectory are skipped entirely – the runtime substitutes the real
    server response at benchmark time.

    Turn structure::

        {
            "messages": [...],   # delta user messages for this turn
            "prompt_tokens": int # total accumulated prompt token count at this turn
        }

    Returns ``None`` if the trajectory does not have enough user messages to fill
    all ``num_turns`` turns.
    """
    user_msgs = [m for m in messages if m['role'] == 'user']

    if not user_msgs:
        return None

    turns: List[Dict] = []
    accumulated_msgs: List[Dict[str, str]] = []

    # ---- Turn 1 ----
    delta, msg_idx = _collect_until(user_msgs, 0, first_turn_length, tokenizer)
    if not delta:
        return None
    accumulated_msgs.extend(delta)
    prompt_tokens = _count_tokens_for_messages(accumulated_msgs, tokenizer)
    turns.append({'messages': delta, 'prompt_tokens': prompt_tokens})

    # ---- Turns 2..num_turns ----
    for _ in range(num_turns - 1):
        if msg_idx >= len(user_msgs):
            return None  # trajectory exhausted before filling all turns
        delta, msg_idx = _collect_until(user_msgs, msg_idx, subsequent_turn_length, tokenizer)
        if not delta:
            return None
        accumulated_msgs.extend(delta)
        prompt_tokens = _count_tokens_for_messages(accumulated_msgs, tokenizer)
        turns.append({'messages': delta, 'prompt_tokens': prompt_tokens})

    return turns


# ---------------------------------------------------------------------------
# Dataset plugin
# ---------------------------------------------------------------------------


class _NullContext:
    """No-op context manager used when multiprocessing is disabled (num_workers=1)."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


@register_dataset('swe_smith')
class SweSmithDatasetPlugin(DatasetPluginBase):
    """Multi-turn dataset plugin backed by SWE-smith-trajectories.

    Two modes:

    * **Pre-built** (``--dataset-path``): Load ``agentic_dataset.json``
      directly.  Each conversation is a list of turn dicts with ``messages``
      (delta) and ``prompt_tokens`` metadata.

    * **Live construction** (no ``--dataset-path``): Stream trajectories from
      ModelScope, apply the same logic as ``build_swe_smith_dataset.py``, and
      yield conversations on-the-fly.  Token-length parameters are taken from
      ``multi_turn_args``.

    The number of turns per conversation is sampled uniformly from
    ``[min_turns, max_turns]`` (``--min-turns`` / ``--max-turns``) for each
    conversation independently.  Each turn is filled with user messages up to
    the configured ``first_turn_length`` / ``subsequent_turn_length`` token
    targets; conversations that run out of messages before filling all turns
    are discarded.

    Parameters from ``multi_turn_args`` that affect live construction:

    * ``first_turn_length``      – target tokens for turn 1
    * ``subsequent_turn_length`` – token growth per subsequent turn
    * ``chars_per_token``        – pre-filter estimate
    * ``num_workers``            – multiprocessing pool size
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_messages(self) -> Iterator[Conversation]:
        """Yield up to ``args.number`` conversations as ``List[Turn]``.

        ``max_tokens`` and ``tool_call_latency`` are left ``None``; swe_smith
        does not specify per-turn output caps or simulated tool waits.
        """
        if self.query_parameters.dataset_path:
            yield from self._load_from_json()
        else:
            yield from self._load_live()

    # ------------------------------------------------------------------
    # Pre-built JSON mode
    # ------------------------------------------------------------------

    def _load_from_json(self) -> Iterator[Conversation]:
        """Load pre-built ``agentic_dataset.json`` and yield conversations."""
        dataset_path = self.query_parameters.dataset_path
        logger.info(f'Loading pre-built dataset from {dataset_path}')

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = data.get('conversations', [])
        if not conversations:
            logger.warning(f'No conversations found in {dataset_path}')
            return

        # Apply offset: rotate conversations list before cycling
        offset = self.query_parameters.dataset_offset
        if offset > 0:
            conversations = conversations[offset:] + conversations[:offset]

        logger.info(f'Loaded {len(conversations)} conversations (offset={offset})')

        for conversation in conversations:
            # Each turn's "messages" list is already the delta (non-assistant).
            messages_per_turn: List[Messages] = [turn.get('messages', []) for turn in conversation]
            if not messages_per_turn:
                continue

            yield [
                Turn(messages=m, is_final=(i == len(messages_per_turn) - 1)) for i, m in enumerate(messages_per_turn)
            ]

    # ------------------------------------------------------------------
    # Live construction mode
    # ------------------------------------------------------------------

    def _load_live(self) -> Iterator[Conversation]:
        """Pull SWE-smith-trajectories from ModelScope and construct conversations."""
        if self.tokenizer is None:
            raise ValueError(
                'Live construction mode requires --tokenizer-path for accurate '
                'token counting.  Please specify --tokenizer-path or provide a pre-built '
                '--dataset-path.'
            )

        mt_args = self.query_parameters.multi_turn_args
        chars_per_token = mt_args.chars_per_token if mt_args else 3.0
        offset = self.query_parameters.dataset_offset
        min_turns = self.query_parameters.min_turns
        max_turns = self.query_parameters.max_turns if self.query_parameters.max_turns is not None else min_turns
        if max_turns < min_turns:
            raise ValueError(f'--max-turns ({max_turns}) must be >= --min-turns ({min_turns})')
        num_workers = mt_args.num_workers if mt_args else 1

        # For pre-filtering, use the upper-bound of first_turn_length as the minimum
        # char threshold (conservative: trajectories need at least that many tokens).
        if mt_args is not None:
            from evalscope.perf.multi_turn_args import _get_range_upper
            first_upper = _get_range_upper(mt_args.first_turn_length)
            subsequent_upper = _get_range_upper(mt_args.subsequent_turn_length)
        else:
            first_upper = 65000
            subsequent_upper = 500

        min_tokens_estimate = first_upper + subsequent_upper * (max_turns - 1)
        min_chars = int(min_tokens_estimate * chars_per_token)

        logger.info(
            f'Live construction: offset={offset}, min_turns={min_turns}, max_turns={max_turns}, '
            f'num_workers={num_workers}, min_chars={min_chars}'
        )

        from modelscope import MsDataset
        dataset = MsDataset.load(_DEFAULT_DATASET_NAME, split=_DEFAULT_SPLIT)

        target_count = max(1, self.query_parameters.number)

        candidates = []
        skipped_short = 0
        skipped_parse = 0

        for row in tqdm(dataset, desc='Scanning dataset', unit='row'):
            raw_messages = row.get('messages', '') if isinstance(row, dict) else row['messages']
            if len(raw_messages) < min_chars:
                skipped_short += 1
                continue
            try:
                messages = _extract_messages(raw_messages)
            except (json.JSONDecodeError, KeyError, TypeError):
                skipped_parse += 1
                continue
            total_chars = sum(len(m['content']) for m in messages if m['role'] == 'user')
            if total_chars < min_chars:
                skipped_short += 1
                continue
            candidates.append(messages)

        logger.info(
            f'Pre-filter: {len(candidates)} candidates '
            f'({skipped_short} too short, {skipped_parse} parse error)'
        )

        np.random.shuffle(candidates)

        # Apply offset
        if offset > 0:
            candidates = candidates[offset:] + candidates[:offset]

        # Pre-sample params per candidate in the main process so different
        # conversations get different token-length targets and turn counts.
        work_items = []
        for messages in candidates:
            sampled = self.get_sampled_multi_turn_params()
            num_turns = int(np.random.randint(min_turns, max_turns + 1))
            work_items.append((
                messages,
                self.tokenizer,
                sampled.get('first_turn_length', 65000),
                sampled.get('subsequent_turn_length', 500),
                num_turns,
            ))

        logger.info(
            f'Building up to {target_count} conversations from {len(work_items)} candidates '
            f'using {num_workers} worker(s)...'
        )

        conversations: List[List[Messages]] = []
        skipped_build = 0

        # Use spawn context to avoid fork-based deadlocks on Linux: the perf runner's
        # parent process contains async loops, logger threads, tqdm monitors, and a
        # possibly pre-warmed HF tokenizer (Rust Rayon threads). fork would inherit
        # these threads' lock states and deadlock on Pool cleanup.
        mp_ctx = multiprocessing.get_context('spawn')
        pool_ctx = (mp_ctx.Pool(num_workers) if num_workers > 1 else None)

        ctx = pool_ctx if pool_ctx is not None else _NullContext()
        with ctx as pool:
            imap = pool.imap if pool is not None else lambda fn, items: map(fn, items)
            with tqdm(total=target_count, desc='Building conversations') as pbar:
                for conv in imap(_build_one, work_items):
                    if conv is None:
                        skipped_build += 1
                    else:
                        conversations.append(conv)
                        pbar.update(1)
                    if len(conversations) >= target_count:
                        if pool is not None:
                            pool.terminate()
                        break

        logger.info(f'Built {len(conversations)} conversations '
                    f'({skipped_build} skipped during build)')

        if conversations:
            all_turns = [len(conv) for conv in conversations]
            all_first_pt = [conv[0]['prompt_tokens'] for conv in conversations]
            all_last_pt = [conv[-1]['prompt_tokens'] for conv in conversations]
            logger.info(
                f'  Turns per conversation : min={min(all_turns)}, '
                f'max={max(all_turns)}, avg={sum(all_turns)/len(all_turns):.1f}'
            )
            logger.info(
                f'  First turn prompt tokens: min={min(all_first_pt)}, '
                f'max={max(all_first_pt)}, avg={sum(all_first_pt)/len(all_first_pt):.0f}'
            )
            logger.info(
                f'  Last  turn prompt tokens: min={min(all_last_pt)}, '
                f'max={max(all_last_pt)}, avg={sum(all_last_pt)/len(all_last_pt):.0f}'
            )

        for conversation in conversations:
            messages_per_turn: List[Messages] = [turn.get('messages', []) for turn in conversation]
            if not messages_per_turn:
                continue
            yield [
                Turn(messages=m, is_final=(i == len(messages_per_turn) - 1)) for i, m in enumerate(messages_per_turn)
            ]
