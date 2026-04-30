"""Multi-turn conversation dataset plugins for evalscope perf.

Plugins
-------
* ``random_multi_turn``          - Synthetic multi-turn conversations (requires tokenizer).
* ``share_gpt_zh_multi_turn``    - Chinese ShareGPT conversations (all turns preserved).
* ``share_gpt_en_multi_turn``    - English ShareGPT conversations (all turns preserved).

Dataset format yielded by ``build_messages()``
----------------------------------------------
Each call to ``build_messages()`` yields a *complete conversation* as a list of
OpenAI-style message dicts.  For random data only user turns are included; for
ShareGPT data the full user+assistant alternation is preserved so that the
multi-turn runner can seed its context with the reference assistant replies.

Example (random_multi_turn):
    [
        {'role': 'user', 'content': 'token sequence turn 1 ...'},
        {'role': 'user', 'content': 'token sequence turn 2 ...'},
    ]

Example (share_gpt_*_multi_turn):
    [
        {'role': 'user',      'content': 'How are you?'},
        {'role': 'assistant', 'content': 'I am fine, thanks!'},
        {'role': 'user',      'content': 'Tell me a joke.'},
        {'role': 'assistant', 'content': 'Why did the chicken...'},
        {'role': 'user',      'content': 'Another one?'},
    ]
"""

import json
import numpy as np
import os
from typing import Any, Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import Message, Messages
from evalscope.perf.plugin.datasets.random_dataset import RandomDatasetPlugin
from evalscope.perf.plugin.datasets.share_gpt import ShareGPTDatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Random multi-turn
# ---------------------------------------------------------------------------


@register_dataset('random_multi_turn')
class RandomMultiTurnDatasetPlugin(RandomDatasetPlugin):
    """Synthetic multi-turn dataset that reuses RandomDatasetPlugin token generation.

    Each yielded item is a **full conversation** (list of user-turn dicts).
    The number of turns per conversation is sampled uniformly from
    ``[min_turns, max_turns]``.  ``max_turns`` is required.

    Produces ``args.number`` conversations upfront (worst-case one turn each
    guarantees enough turn budget for the benchmark runner).
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

        # Read min/max_turns from multi_turn_args if set, otherwise fall back
        # to top-level fields for backward compatibility.
        mt_args = query_parameters.multi_turn_args
        if mt_args is not None:
            self.min_turns_per_conv = max(1, mt_args.min_turns)
            self.max_turns_per_conv = mt_args.max_turns
        else:
            self.min_turns_per_conv = max(1, query_parameters.min_turns)
            self.max_turns_per_conv = query_parameters.max_turns

        if self.max_turns_per_conv is None:
            raise ValueError(
                'random_multi_turn requires --max-turns to be set. '
                'Please specify the maximum number of turns per conversation.'
            )
        if self.max_turns_per_conv < self.min_turns_per_conv:
            raise ValueError(
                f'--max-turns ({self.max_turns_per_conv}) must be >= '
                f'--min-turns ({self.min_turns_per_conv}).'
            )

    def build_messages(self) -> Iterator[List[Messages]]:
        """Yield complete synthetic conversations.

        Each conversation is a ``List[Messages]`` where every ``Messages`` is a
        one-element list ``[{'role': 'user', 'content': <text>}]`` representing
        one turn's delta.  The multi-turn benchmark runner appends the model's
        real response after each turn and passes the growing context to the next.

        Note: ``--tokenize-prompt`` is not supported for multi-turn datasets
        (multi-turn requires the chat/completions endpoint which expects message
        dicts, not raw token-ID lists).  The flag is silently ignored here.
        """
        if self.query_parameters.tokenize_prompt:
            logger.warning(
                '[random_multi_turn] --tokenize-prompt is not supported in multi-turn mode '
                'and will be ignored.  Multi-turn conversations are always sent as message dicts '
                'to the chat/completions endpoint.'
            )
        min_prompt_length = self.query_parameters.min_prompt_length
        max_prompt_length = self.query_parameters.max_prompt_length

        # Total number of conversations to pre-generate.
        # args.number is the total *turn* budget, not the desired conversation
        # count.  Equating the two over-allocates by a factor of ~avg_turns,
        # wasting significant memory when --number is large.
        # Instead, estimate the required conversations from the expected turns
        # per conversation, and keep a small diversity buffer for workers.
        avg_turns = (self.min_turns_per_conv + self.max_turns_per_conv) / 2.0
        n_convs = max(
            self.query_parameters.parallel * 4,  # diversity buffer so workers don't all repeat the same conv
            int(self.number / avg_turns) + 1,  # enough to cover the turn budget
        )

        # Sample per-conversation turn counts
        turn_counts = np.random.randint(
            self.min_turns_per_conv,
            self.max_turns_per_conv + 1,
            size=n_convs,
        )

        # Total individual turn slots across all conversations
        total_turns = int(turn_counts.sum())

        # Pre-sample all input lengths and offsets at once for efficiency
        input_lens = np.random.randint(min_prompt_length, max_prompt_length + 1, size=total_turns)
        global_offset = self.query_parameters.dataset_offset
        offsets = (np.random.randint(0, len(self.allowed_tokens), size=total_turns)
                   + global_offset) % len(self.allowed_tokens)

        turn_slot = 0
        for conv_idx in range(n_convs):
            n_turns = int(turn_counts[conv_idx])
            conversation: List[Messages] = []

            for t in range(n_turns):
                prompt, _, _ = self.generate_token_sequence(
                    input_len=int(input_lens[turn_slot]),
                    offset=int(offsets[turn_slot]),
                    index=turn_slot,
                )
                conversation.append([{'role': 'user', 'content': prompt}])
                turn_slot += 1

            yield conversation


# ---------------------------------------------------------------------------
# ShareGPT multi-turn base
# ---------------------------------------------------------------------------


class ShareGPTMultiTurnBase(ShareGPTDatasetPluginBase):
    """ShareGPT plugin that preserves the full user+assistant alternation.

    Unlike the standard ``ShareGPTDatasetPluginBase``, this plugin does NOT
    strip the trailing assistant turn.  The full conversation is yielded so
    that the multi-turn benchmark runner can:

    1. Correctly count user turns and respect ``max_turns`` limits.
    2. (Future) Optionally replay dataset assistant turns as seeded history.

    In the current implementation the runner discards all dataset assistant
    turns and accumulates only the model's real responses as conversation
    history.

    ``args.max_turns`` limits how many *user* turns to include.  If
    ``max_turns`` is set to N, only the first N user turns (and their
    preceding assistant replies) are included.

    Note on assistant messages in the yielded conversation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The assistant messages from the dataset are included in the yielded
    conversation list for completeness, but the benchmark runner
    (``conversation_worker``) always discards them via ``_extract_user_turns``
    and uses the model's **real** output to build the growing context.  The
    reference assistant content in the dataset is therefore never sent to the
    model as history; only actual model responses are accumulated.
    """

    def _convert_to_openai_messages_full(self, conversation: List[Dict]) -> List[Messages]:
        """Convert swift/sharegpt format to a list of per-turn delta Messages.

        Args:
            conversation: List of dicts with ``'human'`` and ``'assistant'`` keys.

        Returns:
            ``List[Messages]`` where each ``Messages`` is
            ``[{'role': 'user', 'content': human_text}]``.  The dataset
            assistant content is discarded; the benchmark runner fills the
            gaps with real model responses.
        """
        # Read max_turns from multi_turn_args if set, otherwise fall back
        # to top-level field for backward compatibility.
        mt_args = self.query_parameters.multi_turn_args
        if mt_args is not None:
            max_turns = mt_args.max_turns
        else:
            max_turns = self.query_parameters.max_turns
        turns: List[Messages] = []
        user_turn_count = 0

        for turn in conversation:
            human = turn.get('human', '').strip()

            if not human:
                continue

            # Respect max_turns: count user turns and stop when exceeded
            if max_turns is not None and user_turn_count >= max_turns:
                break

            turns.append([{'role': 'user', 'content': human}])
            user_turn_count += 1

        return turns

    def build_messages(self) -> Iterator[List[Messages]]:
        """Yield full conversations as List[Messages] (one Messages per user turn)."""
        if not self.query_parameters.dataset_path:
            from modelscope import dataset_snapshot_download

            local_path = dataset_snapshot_download('swift/sharegpt', allow_patterns=[self.FILE_NAME])
            self.query_parameters.dataset_path = os.path.join(local_path, self.FILE_NAME)

        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            item = json.loads(item)
            conversation = item.get('conversation', [])
            if not conversation:
                continue

            turns = self._convert_to_openai_messages_full(conversation)

            # A valid multi-turn conversation needs at least one turn
            if not turns:
                continue

            # Length filter: check the first user message of the first turn as a proxy
            first_user_content = turns[0][0]['content']
            is_valid, _ = self.check_prompt_length(first_user_content)
            if is_valid:
                yield turns


# ---------------------------------------------------------------------------
# Concrete ShareGPT multi-turn plugins
# ---------------------------------------------------------------------------


@register_dataset('share_gpt_zh_multi_turn')
class ShareGPTZhMultiTurnPlugin(ShareGPTMultiTurnBase):
    """Multi-turn Chinese ShareGPT dataset plugin.

    File: common_zh_70k.jsonl (~70k Chinese conversations)
    Dataset: https://www.modelscope.cn/datasets/swift/sharegpt
    """

    FILE_NAME = 'common_zh_70k.jsonl'

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)


@register_dataset('share_gpt_en_multi_turn')
class ShareGPTEnMultiTurnPlugin(ShareGPTMultiTurnBase):
    """Multi-turn English ShareGPT dataset plugin.

    File: common_en_70k.jsonl (~70k English conversations)
    Dataset: https://www.modelscope.cn/datasets/swift/sharegpt
    """

    FILE_NAME = 'common_en_70k.jsonl'

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
