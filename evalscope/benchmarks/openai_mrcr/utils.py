import json
from difflib import SequenceMatcher
from typing import List, Optional

from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

# Token count bins for MRCR metrics
OPENAI_MRCR_BINS = [
    (4096, 8192),
    (8192, 16384),
    (16384, 32768),
    (32768, 65536),
    (65536, 131072),
    (131072, 262144),
    (262144, 524288),
    (524288, 1048576),
]


def get_token_count(text: str, tik_enc) -> int:
    """Get the token count of a text."""
    return len(tik_enc.encode(text))


def get_chatml_tok_cnt(chat_messages_str: str, tik_enc) -> int:
    """Get the token count of a string in chatml format."""
    try:
        messages = json.loads(chat_messages_str)
        if not isinstance(messages, list):
            return 0
        return sum(get_token_count(m.get('content', ''), tik_enc) for m in messages if isinstance(m, dict))
    except json.JSONDecodeError:
        return 0


def str_to_chat_messages(messages_str: str) -> List[ChatMessage]:
    """Convert a string to a list of chat messages."""
    message_mapping = {
        'system': ChatMessageSystem,
        'user': ChatMessageUser,
        'assistant': ChatMessageAssistant,
    }
    try:
        messages = json.loads(messages_str)
        if not isinstance(messages, list):
            return []
    except json.JSONDecodeError:
        return []

    chat_messages = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get('role')
            content = message.get('content')
            msg_cls = message_mapping.get(role)
            if msg_cls and content is not None:
                chat_messages.append(msg_cls(content=content))
    return chat_messages


def bin_index_for(total_tokens: int, bins=OPENAI_MRCR_BINS) -> int:
    """Return bin index for a total token count using OPENAI_MRCR_BINS.
    First and last bins inclusive both ends, middle bins left-inclusive right-exclusive.
    """
    last = len(bins) - 1
    for i, (l, r) in enumerate(bins):
        if i == 0 or i == last:
            if l <= total_tokens <= r:
                return i
        else:
            if l <= total_tokens < r:
                return i
    return 0  # fallback


def grade(prediction: str, reference: str, random_string_to_prepend: Optional[str]) -> float:
    """
    Compare prediction and reference.
    """
    if random_string_to_prepend:
        if not prediction.startswith(random_string_to_prepend):
            return 0.0
        prediction = prediction.removeprefix(random_string_to_prepend)
        reference = reference.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, prediction, reference).ratio())
