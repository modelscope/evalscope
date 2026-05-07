"""Build a multi-turn benchmarking dataset from SWE-bench/SWE-smith-trajectories.

Each trajectory is converted into a multi-turn conversation.  The number of turns
per conversation is sampled uniformly from [--min-turns, --max-turns], aligning with
the live-load behaviour of the swe_smith dataset plugin.

- Turn 1: collect user messages until the prompt reaches --first-turn-length tokens;
  the last message is truncated if it overshoots.
- Subsequent turns: collect user messages until the incremental token growth reaches
  --subsequent-turn-length tokens; truncate the last message if needed.
- If the trajectory runs out of user messages before filling all sampled turns, the
  conversation is discarded (skip).

Each turn stores only the DELTA user messages for that turn.  At benchmark time the
runtime accumulates the full history across turns:
    turn_1.messages -> send -> get response ->
    [actual_server_response] + turn_2.messages -> send -> ...

Assistant messages in the source trajectory are ignored entirely; the runtime always
appends the real server response as the assistant turn.

The dataset can be used with evalscope perf --dataset swe_smith via
--dataset-path pointing to the generated JSON file.

Usage:
    python examples/perf/build_swe_smith_dataset.py \\
        --model-path Qwen/Qwen2.5-7B-Instruct \\
        --first-turn-length 65000 \\
        --subsequent-turn-length 500 \\
        --min-turns 5 \\
        --max-turns 10 \\
        --number 128 \\
        --output-path agentic_dataset.json \\
        --seed 42 \\
        --num-workers 8
"""

import argparse
import json
import multiprocessing
import numpy as np
import sys
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from evalscope.perf.plugin.datasets.utils import tokenize_chat_messages

# Dataset constants (same as swe_smith.py plugin)
_DEFAULT_DATASET_NAME = 'SWE-bench/SWE-smith-trajectories'
_DEFAULT_SPLIT = 'tool'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build multi-turn agentic benchmark dataset from SWE-smith trajectories.'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model path for tokenizer (used for accurate token counting)',
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=_DEFAULT_DATASET_NAME,
        help='ModelScope dataset name',
    )
    parser.add_argument(
        '--split',
        type=str,
        default=_DEFAULT_SPLIT,
        help='Dataset split to use (tool, xml, ticks)',
    )
    parser.add_argument(
        '--first-turn-length',
        type=int,
        default=65000,
        help='Target token length for the first turn prompt (default: 65000)',
    )
    parser.add_argument(
        '--subsequent-turn-length',
        type=int,
        default=500,
        help='Target token growth per subsequent turn (default: 500)',
    )
    parser.add_argument(
        '--min-turns',
        type=int,
        default=1,
        help='Minimum number of turns per conversation (default: 1)',
    )
    parser.add_argument(
        '--max-turns',
        type=int,
        default=None,
        help='Maximum number of turns per conversation (default: same as --min-turns). '
             'The actual turn count for each conversation is sampled uniformly from '
             '[min_turns, max_turns]. Trajectories with fewer user messages than the '
             'sampled turn count are discarded.',
    )
    parser.add_argument(
        '--number',
        type=int,
        default=128,
        help='Number of conversations to generate (default: 128)',
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='outputs/agentic_dataset.json',
        help='Output file path (default: agentic_dataset.json)',
    )
    parser.add_argument(
        '--chars-per-token',
        type=float,
        default=3.0,
        help='Chars per token estimate for pre-filtering (default: 3.0)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: number of CPU cores)',
    )
    return parser.parse_args()


def extract_messages(raw_messages: str) -> List[Dict[str, str]]:
    """Parse the messages JSON string and normalize to [{role, content}, ...]."""
    parsed = json.loads(raw_messages)
    messages = []
    for msg in parsed:
        role = msg.get('role', '')
        content = msg.get('content', '')
        # Handle content that is a list of content blocks
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
        # Normalize role: tool -> user (for chat template compatibility)
        if role == 'tool':
            role = 'user'
        if role not in ('system', 'user', 'assistant'):
            continue
        messages.append({'role': role, 'content': content})
    return messages


def count_tokens_for_messages(messages: List[Dict[str, str]], tokenizer) -> int:
    """Count tokens for a list of chat messages using tokenize_chat_messages."""
    if not messages:
        return 0
    return len(tokenize_chat_messages(tokenizer, messages))


def _bare_encode(text: str, tokenizer) -> List[int]:
    """Encode text to token ids without special tokens."""
    return tokenizer.encode(text, add_special_tokens=False)


def truncate_message_content(
    message: Dict[str, str],
    tokens_needed: int,
    tokenizer,
) -> Dict[str, str]:
    """Truncate a message's content so it occupies at most tokens_needed bare tokens.

    Uses raw encode (no chat template) to slice the content token ids, then decodes
    back to text.  The caller should use count_tokens_for_messages on the full
    accumulated list for an accurate prompt_tokens value afterwards.
    """
    content_tokens = _bare_encode(message['content'], tokenizer)
    truncated_content = tokenizer.decode(content_tokens[:tokens_needed], skip_special_tokens=True)
    return {'role': message['role'], 'content': truncated_content}


def _encode_len(text: str, tokenizer) -> int:
    """Fast bare-text token count (no chat template overhead)."""
    return len(_bare_encode(text, tokenizer))


def collect_until(
    user_msgs: List[Dict[str, str]],
    start_idx: int,
    target_tokens: int,
    tokenizer,
    accumulated_before: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], int]:
    """Collect user messages from start_idx until the prompt grows by ~target_tokens.

    Uses fast bare-text token counting (no chat template) to estimate the
    incremental size of each message.  A message that fits exactly within the
    remaining budget is collected whole.  The last message is truncated only
    when it would *exceed* the budget (``remaining > 0``); if the budget is
    already exhausted the message is left unconsumed (``idx`` not advanced).
    The caller is responsible for computing the final accurate prompt_tokens via
    count_tokens_for_messages after the turn is assembled.

    Args:
        user_msgs: Full list of user-only messages for this trajectory.
        start_idx: Index in user_msgs to start collecting from.
        target_tokens: Desired incremental token growth (bare-token estimate).
        tokenizer: Tokenizer used for token counting.
        accumulated_before: Messages already in the prompt before this turn
                            (used only for the truncation fallback; not tokenized here).

    Returns:
        (delta, new_idx) where delta is the list of collected messages and
        new_idx is the index of the next unconsumed message.
    """
    delta: List[Dict[str, str]] = []
    accumulated_bare = 0  # running bare-token count of delta messages
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
            # This message would overshoot – truncate to the remaining budget.
            remaining = target_tokens - accumulated_bare
            if remaining <= 0:
                # Budget already exhausted; do NOT consume this message.
                break
            truncated = truncate_message_content(msg, remaining, tokenizer)
            delta.append(truncated)
            idx += 1
            break

    return delta, idx



def build_conversation(
    messages: List[Dict[str, str]],
    tokenizer,
    first_turn_length: int,
    subsequent_turn_length: int,
    num_turns: int,
) -> Optional[List[Dict]]:
    """Build a multi-turn conversation from a trajectory.

    Only user messages from the trajectory are used.  Each turn stores the DELTA
    user messages for that turn (not the full history).  assistant messages in the
    source trajectory are skipped entirely – the runtime will substitute the real
    server response at benchmark time.

    Turn structure:
        {
            "messages": [...],   # delta user messages for this turn
            "prompt_tokens": int # total accumulated prompt token count at this turn
        }

    Returns None if the trajectory does not have enough user messages to fill
    all num_turns turns.
    """
    # Extract only user messages (tool responses are already normalised to user
    # by extract_messages; skip system/assistant here).
    user_msgs = [m for m in messages if m['role'] == 'user']

    if not user_msgs:
        return None

    turns: List[Dict] = []
    accumulated_msgs: List[Dict[str, str]] = []  # flat list of all delta messages so far

    # ---- Turn 1 ----
    delta, msg_idx = collect_until(user_msgs, 0, first_turn_length, tokenizer, [])
    if not delta:
        return None
    accumulated_msgs.extend(delta)
    prompt_tokens = count_tokens_for_messages(accumulated_msgs, tokenizer)
    turns.append({'messages': delta, 'prompt_tokens': prompt_tokens})

    # ---- Turns 2..num_turns ----
    for turn_idx in range(num_turns - 1):
        if msg_idx >= len(user_msgs):
            return None  # trajectory exhausted before filling all turns
        delta, msg_idx = collect_until(
            user_msgs, msg_idx, subsequent_turn_length, tokenizer, accumulated_msgs
        )
        if not delta:
            return None
        accumulated_msgs.extend(delta)
        prompt_tokens = count_tokens_for_messages(accumulated_msgs, tokenizer)
        turns.append({'messages': delta, 'prompt_tokens': prompt_tokens})

    return turns


def _build_one(args_tuple):
    """Worker function for multiprocessing."""
    (
        messages,
        tokenizer,
        first_turn_length,
        subsequent_turn_length,
        num_turns,
    ) = args_tuple
    return build_conversation(
        messages,
        tokenizer,
        first_turn_length,
        subsequent_turn_length,
        num_turns,
    )


def main():
    args = parse_args()

    np.random.seed(args.seed)

    # Resolve min/max turns (aligns with live-load swe_smith plugin semantics)
    min_turns = args.min_turns
    max_turns = args.max_turns if args.max_turns is not None else min_turns
    if max_turns < min_turns:
        print(f'Error: --max-turns ({max_turns}) must be >= --min-turns ({min_turns})')
        sys.exit(1)

    print(f'Loading tokenizer from {args.model_path}...')
    from modelscope import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f'Loading dataset {args.dataset_name} (split: {args.split})...')
    from modelscope import MsDataset
    dataset = MsDataset.load(args.dataset_name, split=args.split)

    # Pre-filter: need enough chars to fill max_turns turns (upper bound)
    # total ≈ first_turn_length + subsequent_turn_length * (max_turns - 1)
    min_tokens_estimate = args.first_turn_length + args.subsequent_turn_length * (max_turns - 1)
    min_chars = int(min_tokens_estimate * args.chars_per_token)
    print(
        f'Pre-filtering: require >= {min_chars} chars '
        f'(~{min_tokens_estimate} tokens = {args.first_turn_length} + '
        f'{args.subsequent_turn_length} x {max_turns - 1} turns, '
        f'at {args.chars_per_token} chars/token)'
    )

    candidates = []
    skipped_short = 0
    skipped_parse = 0
    for row in tqdm(dataset, desc='Pre-filtering'):
        raw_messages = row.get('messages', '') if isinstance(row, dict) else row['messages']
        # Quick char-length check before parsing JSON
        if len(raw_messages) < min_chars:
            skipped_short += 1
            continue
        try:
            messages = extract_messages(raw_messages)
        except (json.JSONDecodeError, KeyError, TypeError):
            skipped_parse += 1
            continue
        # Check total content length of user messages only
        total_chars = sum(len(m['content']) for m in messages if m['role'] == 'user')
        if total_chars < min_chars:
            skipped_short += 1
            continue
        candidates.append(messages)

    print(
        f'Pre-filter: {len(candidates)} candidates, '
        f'{skipped_short} skipped (too short), '
        f'{skipped_parse} skipped (parse error)'
    )

    np.random.shuffle(candidates)

    if len(candidates) < args.number:
        print(
            f'Error: Only {len(candidates)} candidates pass pre-filter, '
            f'but {args.number} conversations requested. '
            f'Try a smaller --first-turn-length / --subsequent-turn-length / --min-turns / --max-turns '
            f'or use additional splits.'
        )
        sys.exit(1)

    # Pre-sample per-candidate num_turns (aligns with live-load behaviour)
    num_workers = args.num_workers or multiprocessing.cpu_count()
    work_items = [
        (
            msgs,
            tokenizer,
            args.first_turn_length,
            args.subsequent_turn_length,
            int(np.random.randint(min_turns, max_turns + 1)),
        )
        for msgs in candidates
    ]

    print(
        f'Building conversations with {num_workers} workers '
        f'(turns sampled from [{min_turns}, {max_turns}])...'
    )
    conversations = []
    skipped_build = 0
    # Use spawn context to avoid fork-based deadlocks on Linux.
    mp_ctx = multiprocessing.get_context('spawn')
    with mp_ctx.Pool(num_workers) as pool:
        with tqdm(total=args.number, desc='Building conversations') as pbar:
            for conv in pool.imap(_build_one, work_items):
                if conv is None:
                    skipped_build += 1
                else:
                    conversations.append(conv)
                    pbar.update(1)
                    if len(conversations) >= args.number:
                        pool.terminate()
                        break

    print(f'\nBuilt {len(conversations)} conversations ({skipped_build} skipped during build)')

    if len(conversations) < args.number:
        print(
            f'Warning: Only built {len(conversations)} conversations '
            f'({skipped_build} failed), but {args.number} requested.'
        )

    # Print statistics
    if not conversations:
        print('Error: No conversations were built. Check your dataset and parameters.')
        return

    all_turns = [len(conv) for conv in conversations]
    all_first_turn_tokens = [conv[0]['prompt_tokens'] for conv in conversations]
    all_last_turn_tokens = [conv[-1]['prompt_tokens'] for conv in conversations]
    print(
        f'  Turns per conversation: min={min(all_turns)}, max={max(all_turns)}, '
        f'avg={sum(all_turns)/len(all_turns):.1f}'
    )
    print(
        f'  First turn prompt tokens: min={min(all_first_turn_tokens)}, '
        f'max={max(all_first_turn_tokens)}, '
        f'avg={sum(all_first_turn_tokens)/len(all_first_turn_tokens):.0f}'
    )
    print(
        f'  Last turn prompt tokens: min={min(all_last_turn_tokens)}, '
        f'max={max(all_last_turn_tokens)}, '
        f'avg={sum(all_last_turn_tokens)/len(all_last_turn_tokens):.0f}'
    )

    # Save dataset
    output = {
        'metadata': {
            'model_path': args.model_path,
            'dataset_name': args.dataset_name,
            'split': args.split,
            'first_turn_length': args.first_turn_length,
            'subsequent_turn_length': args.subsequent_turn_length,
            'min_turns': min_turns,
            'max_turns': max_turns,
            'num_conversations': len(conversations),
        },
        'conversations': conversations,
    }

    print(f'Saving to {args.output_path}...')
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print('Done.')


if __name__ == '__main__':
    main()
