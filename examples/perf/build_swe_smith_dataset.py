"""Build a multi-turn benchmarking dataset from SWE-bench/SWE-smith-trajectories.

Each trajectory is converted into a multi-turn conversation where:
- Turn 1: accumulate messages until prompt reaches --first-turn-length tokens
- Subsequent turns: append server response + accumulate messages until prompt grows
  by --subsequent-turn-length tokens
- Stop when total context reaches --max-context-length or trajectory is exhausted

Each turn stores only the DELTA messages (new messages to append), not the full
history. At benchmark time, the full prompt is built incrementally:
    turn_1.messages -> send -> get response ->
    [response_as_assistant_msg] + turn_2.messages -> send -> ...

The dataset can be used with evalscope perf --dataset swe_smith via
--dataset-path pointing to the generated JSON file.

Usage:
    python examples/perf/build_swe_smith_dataset.py \\
        --model-path Qwen/Qwen2.5-7B-Instruct \\
        --first-turn-length 8192 \\
        --subsequent-turn-length 1024 \\
        --max-context-length 12000 \\
        --output-length 512 \\
        --num-conversations 128 \\
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
        '--max-context-length',
        type=int,
        default=75000,
        help='Maximum total context length in tokens (default: 75000)',
    )
    parser.add_argument(
        '--output-length',
        type=int,
        default=300,
        help='Output token length for each turn (default: 300)',
    )
    parser.add_argument(
        '--num-conversations',
        type=int,
        default=128,
        help='Number of conversations to generate (default: 128)',
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='agentic_dataset.json',
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
    """Count tokens for a list of chat messages using apply_chat_template."""
    if not messages:
        return 0
    token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return len(token_ids)


def count_tokens_for_text(text: str, tokenizer) -> int:
    """Count tokens for a plain text string."""
    return len(tokenizer.encode(text))


def truncate_message_content(
    message: Dict[str, str],
    tokens_needed: int,
    tokenizer,
) -> Optional[Dict[str, str]]:
    """Truncate a message's content to approximately tokens_needed tokens.

    Returns the truncated message, or None if tokens_needed <= 0.
    """
    if tokens_needed <= 0:
        return None
    content_tokens = tokenizer.encode(message['content'], add_special_tokens=False)
    truncated_content = tokenizer.decode(content_tokens[:tokens_needed], skip_special_tokens=True)
    return {'role': message['role'], 'content': truncated_content}


def _accumulate_messages(
    messages: List[Dict[str, str]],
    msg_idx: int,
    msg_char_offset: int,
    current_messages: List[Dict[str, str]],
    current_tokens: int,
    target_tokens: int,
    tokenizer,
) -> Tuple[List[Dict[str, str]], int, int, int]:
    """Append messages from the trajectory until reaching target_tokens.

    Supports partial consumption of large messages: if a message is too long,
    only the portion needed to reach target_tokens is consumed, and
    msg_char_offset tracks where to resume in the next call.

    Rules:
    - If a message would push past target, truncate it to hit exactly target.
    - Never end on an assistant message (drop trailing assistant from delta).

    Returns:
        (delta_messages, new_total_tokens, new_msg_idx, new_msg_char_offset)
    """
    delta = []

    while msg_idx < len(messages):
        msg = messages[msg_idx]
        # If we have a partial offset into this message, use the remainder
        if msg_char_offset > 0:
            msg = {'role': msg['role'], 'content': msg['content'][msg_char_offset:]}
            if not msg['content']:
                msg_idx += 1
                msg_char_offset = 0
                continue

        trial_messages = current_messages + delta + [msg]
        trial_tokens = count_tokens_for_messages(trial_messages, tokenizer)

        if trial_tokens < target_tokens:
            delta.append(msg)
            current_tokens = trial_tokens
            msg_idx += 1
            msg_char_offset = 0
        else:
            # Would meet or exceed target — need to truncate or stop
            if msg['role'] == 'assistant':
                # Don't end on assistant, skip it
                msg_idx += 1
                msg_char_offset = 0
                continue
            if trial_tokens == target_tokens:
                delta.append(msg)
                current_tokens = trial_tokens
                msg_idx += 1
                msg_char_offset = 0
            else:
                # Truncate this message to approximately hit target
                tokens_needed = target_tokens - current_tokens
                truncated = truncate_message_content(msg, tokens_needed, tokenizer)
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

    # Ensure we don't end on an assistant message
    while delta and delta[-1]['role'] == 'assistant':
        delta.pop()

    if delta:
        current_tokens = count_tokens_for_messages(current_messages + delta, tokenizer)

    return delta, current_tokens, msg_idx, msg_char_offset


def build_conversation(
    messages: List[Dict[str, str]],
    tokenizer,
    first_turn_length: int,
    subsequent_turn_length: int,
    max_context_length: int,
    output_length: int,
) -> Optional[List[Dict]]:
    """Build a multi-turn conversation from a trajectory's messages.

    Returns a list of turns, each turn stores only the DELTA messages for that
    turn (not the full history). At benchmark time, the full prompt is built by
    concatenating: turn_1.messages + server_resp_1 + turn_2.messages + ...

    Each turn is:
        {
            "messages": [...],       # delta messages to append for this turn
            "prompt_tokens": int,    # total token count of the full prompt at this turn
            "output_tokens": int,    # expected output token count
        }

    Returns None if the trajectory is too short.
    """
    turns = []
    accumulated = []  # Full accumulated chat messages (for token counting)
    msg_idx = 0
    msg_char_offset = 0  # Partial offset into current message
    current_tokens = 0

    # Build first turn
    delta, current_tokens, msg_idx, msg_char_offset = _accumulate_messages(
        messages,
        msg_idx,
        msg_char_offset,
        accumulated,
        current_tokens,
        first_turn_length,
        tokenizer,
    )

    if current_tokens < first_turn_length * 0.8:
        return None

    accumulated.extend(delta)
    turns.append(
        {
            'messages': delta,
            'prompt_tokens': current_tokens,
            'output_tokens': output_length,
        }
    )

    # Pre-generate a fake response that tokenizes to output_length tokens
    fake_response_text = make_fake_response(tokenizer, output_length)

    # Build subsequent turns
    while msg_idx < len(messages) and current_tokens + output_length < max_context_length:
        # Add a fake response for server response to accumulated for accurate token counting
        # (will be replaced by actual server response at benchmark time)
        fake_response = {'role': 'assistant', 'content': fake_response_text}
        accumulated.append(fake_response)
        current_tokens = count_tokens_for_messages(accumulated, tokenizer)

        if current_tokens + output_length >= max_context_length:
            break

        # Skip the first assistant message to avoid consecutive assistant
        # messages (fake server response is already assistant role).
        # Only skip if we're at a message boundary (not mid-message).
        if (
            msg_char_offset == 0
            and msg_idx < len(messages)
            and messages[msg_idx]['role'] == 'assistant'
        ):
            msg_idx += 1

        if msg_idx >= len(messages):
            break

        target = current_tokens + subsequent_turn_length
        if target + output_length > max_context_length:
            break

        delta, current_tokens, msg_idx, msg_char_offset = _accumulate_messages(
            messages,
            msg_idx,
            msg_char_offset,
            accumulated,
            current_tokens,
            target,
            tokenizer,
        )

        if not delta or abs(current_tokens - target) > 5:
            # Trajectory exhausted or couldn't fill the turn — stop
            break

        accumulated.extend(delta)
        turns.append(
            {
                'messages': delta,
                'prompt_tokens': current_tokens,
                'output_tokens': output_length,
            }
        )

    # Expected turns from ideal formula; drift may cost at most one turn.
    remaining = max_context_length - output_length - first_turn_length
    step = subsequent_turn_length + output_length
    expected_turns = 1 + remaining // step

    if len(turns) in (expected_turns, expected_turns - 1):
        return turns
    else:
        return None


def make_fake_response(tokenizer, num_tokens: int) -> str:
    """Generate a fake response by decoding num_tokens token IDs.

    This approximates what the server produces with ignore_eos=True +
    max_new_tokens. The decode/encode roundtrip may not be perfectly
    lossless, but the small difference matches real benchmark behavior
    where the actual server response is arbitrary generated text.
    """
    return tokenizer.decode(list(range(1000, 1000 + num_tokens)), skip_special_tokens=True)


def _build_one(args_tuple):
    """Worker function for multiprocessing."""
    (
        messages,
        tokenizer,
        first_turn_length,
        subsequent_turn_length,
        max_context_length,
        output_length,
    ) = args_tuple
    return build_conversation(
        messages,
        tokenizer,
        first_turn_length,
        subsequent_turn_length,
        max_context_length,
        output_length,
    )


def main():
    args = parse_args()

    np.random.seed(args.seed)

    print(f'Loading tokenizer from {args.model_path}...')
    from modelscope import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f'Loading dataset {args.dataset_name} (split: {args.split})...')
    from modelscope import MsDataset
    dataset = MsDataset.load(args.dataset_name, split=args.split)

    # Pre-filter: estimate total chars needed for max_context_length
    min_chars = int(args.max_context_length * args.chars_per_token)
    print(
        f'Pre-filtering: require >= {min_chars} chars '
        f'(~{args.max_context_length} tokens at {args.chars_per_token} chars/token)'
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
        # Check total content length
        total_chars = sum(len(m['content']) for m in messages)
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

    if len(candidates) < args.num_conversations:
        print(
            f'Error: Only {len(candidates)} candidates pass pre-filter, '
            f'but {args.num_conversations} conversations requested. '
            f'Try a smaller --max-context-length or use additional splits.'
        )
        sys.exit(1)

    # Build conversations in parallel, process all candidates until we have enough
    num_workers = args.num_workers or multiprocessing.cpu_count()
    work_items = [
        (
            msgs,
            tokenizer,
            args.first_turn_length,
            args.subsequent_turn_length,
            args.max_context_length,
            args.output_length,
        )
        for msgs in candidates
    ]

    print(f'Building conversations with {num_workers} workers...')
    conversations = []
    skipped_build = 0
    with multiprocessing.Pool(num_workers) as pool:
        for conv in tqdm(
            pool.imap(_build_one, work_items),
            total=len(work_items),
            desc='Building conversations',
        ):
            if conv is None:
                skipped_build += 1
            else:
                conversations.append(conv)
                if len(conversations) >= args.num_conversations:
                    pool.terminate()
                    break

    print(f'\nBuilt {len(conversations)} conversations ({skipped_build} skipped during build)')

    if len(conversations) < args.num_conversations:
        print(
            f'Warning: Only built {len(conversations)} conversations '
            f'({skipped_build} failed), but {args.num_conversations} requested.'
        )

    # Print statistics
    all_turns = [len(conv) for conv in conversations]
    all_first_turn_tokens = [conv[0]['prompt_tokens'] for conv in conversations]
    all_last_turn_tokens = [conv[-1]['prompt_tokens'] for conv in conversations]
    print(
        f'  Turns per conversation: min={min(all_turns)}, max={max(all_turns)}, '
        f'avg={sum(all_turns)/len(all_turns):.1f}'
    )
    print(
        f'  First turn tokens: min={min(all_first_turn_tokens)}, '
        f'max={max(all_first_turn_tokens)}, '
        f'avg={sum(all_first_turn_tokens)/len(all_first_turn_tokens):.0f}'
    )
    print(
        f'  Last turn tokens: min={min(all_last_turn_tokens)}, '
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
            'max_context_length': args.max_context_length,
            'output_length': args.output_length,
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
