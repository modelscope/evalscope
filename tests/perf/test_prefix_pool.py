"""Shared prefix pools, tested without model downloads or inference services."""

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from evalscope.perf.arguments import Arguments, add_argument
from evalscope.perf.benchmark import get_requests
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.datasets.multi_turn import RandomMultiTurnDatasetPlugin
from evalscope.perf.plugin.datasets.random_dataset import RandomDatasetPlugin
from evalscope.perf.plugin.datasets.random_vl_dataset import RandomVLDatasetPlugin
from evalscope.perf.plugin.datasets.trie import TrieAgenticCodingPlugin
from evalscope.perf.plugin.datasets.utils import gen_prompt_decode_to_target_len, tokenize_chat_messages


def _args(**kwargs: Any) -> Arguments:
    params = dict(
        model='test',
        dataset='random',
        number=8,
        num_workers=1,
        min_prompt_length=8,
        max_prompt_length=8,
        prefix_length=3,
        prefix_count=7,
        apply_chat_template=False,
    )
    params.update(kwargs)
    return Arguments(**params)


@pytest.fixture
def tokenizer_path(tmp_path: Path) -> str:
    backend = Tokenizer(models.BPE({char: i for i, char in enumerate('abcdefghijklmnop')}, []))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, clean_up_tokenization_spaces=False)
    tokenizer.chat_template = "{{ 'a' + messages[0]['content'] + 'b' }}"
    tokenizer.save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize('value', [0, -1, 1.5, 2.0, True, '7', None])
def test_prefix_count_requires_positive_integer(value: Any) -> None:
    with pytest.raises(ValidationError, match='prefix_count'):
        _args(prefix_count=value)


@pytest.mark.parametrize(
    'changes,match',
    [
        ({'prefix_length': 0}, 'prefix-length > 0'),
        ({'prefix_length': -1}, 'prefix-length > 0'),
        ({'dataset': 'random_vl'}, 'only with --dataset random'),
        ({'dataset': 'random_multi_turn'}, 'only with --dataset random'),
        ({'dataset': 'trie_agentic_coding'}, 'only with --dataset random'),
        ({'dataset': 'trie_code_qa'}, 'only with --dataset random'),
        ({'dataset': 'trie_office_work'}, 'only with --dataset random'),
        ({'dataset': 'openqa'}, 'only with --dataset random'),
        ({'prompt': 'literal prompt'}, 'conflicts with --prompt'),
    ],
)
def test_prefix_pool_conflicts(changes: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        _args(**changes)


def test_cli_dict_and_serialization() -> None:
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = Arguments.from_args(
        parser.parse_args(
            [
                '--model',
                'test',
                '--dataset',
                'random',
                '--prefix-length',
                '3',
                '--prefix-count',
                '7',
            ]
        )
    )
    assert args.prefix_count == 7
    assert args.to_dict()['prefix_count'] == 7
    assert Arguments.from_dict(args.to_dict()).prefix_count == 7
    assert Arguments.from_args(parser.parse_args(['--model', 'test'])).prefix_count == 1
    with pytest.raises(SystemExit):
        parser.parse_args(['--prefix-count', '1.5'])


@pytest.mark.parametrize('number,warmup', [(100, 0), (98, 0), (3, 0), (100, 5)])
def test_round_robin_and_warmup(tokenizer_path: str, number: int, warmup: int) -> None:
    args = _args(
        tokenizer_path=tokenizer_path, number=number, warmup_num=warmup, tokenize_prompt=True, dataset_offset=19
    )
    np.random.seed(42)
    plugin = RandomDatasetPlugin(args)
    pool = deepcopy(plugin.prefix_pool)
    messages = list(plugin.build_messages())
    assert len(messages) == number + warmup
    assert len({tuple(prefix) for prefix in pool}) == 7
    assert all(len(prefix) == 3 for prefix in pool)
    assert all(message[:3] == pool[i % 7] and len(message) == 11 for i, message in enumerate(messages))
    counts = Counter(tuple(message[:3]) for message in messages[warmup:])
    usages = [counts[tuple(prefix)] for prefix in pool]
    assert max(usages) - min(usages) <= 1
    if number == 100:
        assert sorted(usages) == [14] * 5 + [15] * 2
    if number == 98:
        assert usages == [14] * 7
    for start in range(len(messages)):
        for end in range(start + 1, len(messages) + 1):
            counts = Counter(tuple(message[:3]) for message in messages[start:end])
            usages = [counts[tuple(prefix)] for prefix in pool]
            assert max(usages) - min(usages) <= 1
    messages[0][0] = -1
    assert plugin.prefix_pool == pool

    class Api:
        def build_request(self, messages: list) -> dict:
            return {'prompt': messages}

    async def collect() -> list:
        return [item async for item in get_requests(args, Api())]

    np.random.seed(42)
    requests = asyncio.run(collect())
    assert [is_warmup for _, is_warmup in requests] == [True] * warmup + [False] * number
    assert [request['prompt'][:3] for request, _ in requests] == [pool[i % 7] for i in range(args.total_count)]


@pytest.mark.parametrize('tokenize,chat', [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize('prefix_count', [1, 7])
def test_real_spawn_matches_serial_and_reproducibility(
    tokenizer_path: str, tokenize: bool, chat: bool, prefix_count: int
) -> None:
    args = _args(
        tokenizer_path=tokenizer_path,
        tokenize_prompt=tokenize,
        apply_chat_template=chat,
        warmup_num=3,
        num_workers=2,
        prefix_count=prefix_count,
    )
    np.random.seed(123)
    plugin = RandomDatasetPlugin(args)
    pool = deepcopy(plugin.prefix_pool)
    rng_state = np.random.get_state()
    serial = list(plugin.build_messages())
    np.random.set_state(rng_state)
    parallel = plugin.build_messages_parallel(args.total_count, workers=2)
    assert parallel == serial
    assert plugin.prefix_pool == pool
    np.random.seed(123)
    repeated = RandomDatasetPlugin(args)
    assert repeated.prefix_pool == pool
    assert list(repeated.build_messages()) == serial
    api = OpenaiPlugin(args)
    prefixes = []
    for i, message in enumerate(serial):
        request = api.build_request(message)
        assert request['messages' if chat and not tokenize else 'prompt'] == message
        if tokenize:
            ids = message
            prefix = ids[:3]
            assert len(ids) == 11
        else:
            text = message[0]['content'] if chat else message
            ids = plugin.tokenizer.encode(text, add_special_tokens=False)
            prefix = ids[:3]
            assert len(ids) == (9 if chat else 11)
            if chat:
                templated = tokenize_chat_messages(plugin.tokenizer, message)
                assert len(templated) == 11
                assert templated[1:4] == pool[i % prefix_count]
        assert prefix == pool[i % prefix_count]
        prefixes.append(tuple(prefix))
    assert len(set(prefixes)) == prefix_count


def test_each_instance_owns_its_pool(tokenizer_path: str) -> None:
    args = _args(tokenizer_path=tokenizer_path)
    np.random.seed(123)
    first = RandomDatasetPlugin(args)
    second = RandomDatasetPlugin(args)
    assert first.prefix_pool != second.prefix_pool
    second_pool = deepcopy(second.prefix_pool)
    first.prefix_pool[0][0] = -1
    assert second.prefix_pool == second_pool


def test_prefix_count_is_not_a_dataset_args_entry(tokenizer_path: str) -> None:
    with pytest.raises(ValidationError, match='prefix_count'):
        RandomDatasetPlugin(_args(tokenizer_path=tokenizer_path, dataset_args={'prefix_count': 7}))


@pytest.mark.parametrize('prefix_length', [0, 3])
@pytest.mark.parametrize('tokenize,chat', [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize('merge', [False, True])
def test_default_matches_legacy_seeded_algorithm(
    tokenizer_path: str, prefix_length: int, tokenize: bool, chat: bool, merge: bool
) -> None:
    if merge:
        vocab = {char: i for i, char in enumerate('abcdefghijklmnop')}
        vocab['ab'] = 16
        backend = Tokenizer(models.BPE(vocab, [('a', 'b')]))
        backend.decoder = decoders.Fuse()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, clean_up_tokenization_spaces=False)
        tokenizer.chat_template = "{{ 'a' + messages[0]['content'] + 'b' }}"
        tokenizer.save_pretrained(tokenizer_path)
    args = _args(
        tokenizer_path=tokenizer_path,
        prefix_count=1,
        prefix_length=prefix_length,
        tokenize_prompt=tokenize,
        apply_chat_template=chat,
        dataset_offset=5,
        warmup_num=2,
    )
    np.random.seed(99)
    plugin = RandomDatasetPlugin(args)
    actual = list(plugin.build_messages())
    state_after = np.random.get_state()
    # Reference algorithm from the pre-pool implementation, independent of pool selection.
    np.random.seed(99)
    tokens = plugin.allowed_tokens
    prefix = tokens[np.random.randint(0, len(tokens), size=prefix_length)].tolist() if prefix_length else []
    template_len = len(tokenize_chat_messages(plugin.tokenizer, [{'role': 'user', 'content': ''}]))
    body_len = 8 - template_len if chat and not tokenize else 8
    lengths = np.random.randint(body_len, body_len + 1, size=args.total_count)
    offsets = (np.random.randint(0, len(tokens), size=args.total_count) + 5) % len(tokens)
    seeds = None if tokenize else np.random.randint(0, np.iinfo(np.uint32).max, size=args.total_count, dtype=np.uint32)
    expected = []
    for index, (length, offset) in enumerate(zip(lengths, offsets)):
        sequence = prefix + tokens[(offset + index + np.arange(length)) % len(tokens)].tolist()
        if tokenize:
            expected.append(sequence)
            continue
        rng = np.random.get_state()
        np.random.seed(int(seeds[index]))
        prompt, _, _ = gen_prompt_decode_to_target_len(
            plugin.tokenizer, sequence, prefix_length + int(length), allowed_tokens=tokens
        )
        np.random.set_state(rng)
        expected.append([{'role': 'user', 'content': prompt}] if chat else prompt)
    assert actual == expected
    reference_state = np.random.get_state()
    assert state_after[0] == reference_state[0]
    np.testing.assert_array_equal(state_after[1], reference_state[1])
    assert state_after[2:] == reference_state[2:]


def test_full_small_space_and_collisions(tokenizer_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = RandomDatasetPlugin(_args(tokenizer_path=tokenizer_path, prefix_length=2, prefix_count=4))
    plugin.allowed_tokens = np.array([4, 9])
    monkeypatch.setattr(plugin, 'get_random_inputs', lambda length: [4] * length)
    pool = plugin._create_prefix_pool()
    assert pool == [[4, 4], [4, 9], [9, 4], [9, 9]]
    plugin.query_parameters.prefix_count = 5
    with pytest.raises(ValueError, match='distinct prefix space'):
        plugin._create_prefix_pool()
    plugin.allowed_tokens = np.array([], dtype=int)
    with pytest.raises(ValueError, match='distinct prefix space'):
        plugin._create_prefix_pool()


@pytest.mark.parametrize(
    'plugin_class,dataset',
    [
        (RandomVLDatasetPlugin, 'random_vl'),
        (RandomMultiTurnDatasetPlugin, 'random_multi_turn'),
        (TrieAgenticCodingPlugin, 'trie_agentic_coding'),
    ],
)
def test_inherited_default_prefix_behavior(tokenizer_path: str, plugin_class: type, dataset: str) -> None:
    args = _args(tokenizer_path=tokenizer_path, dataset=dataset, prefix_count=1, max_turns=2, apply_chat_template=True)
    # The benchmark runner resolves sweep fields to scalars for each run.
    args.parallel = 1
    np.random.seed(12)
    plugin = plugin_class(args)
    np.random.seed(12)
    expected = plugin.get_random_inputs(3)
    assert plugin.prefix_pool == [expected]
    assert plugin.generate_token_ids_only(8, 2, 0)[:3] == expected
    assert plugin.generate_token_ids_only(8, 2, 7)[:3] == expected
    text, length, mismatch = plugin.generate_token_sequence(8, 2, 7)
    assert plugin.tokenizer.encode(text, add_special_tokens=False)[:3] == expected
    assert length == 11 and mismatch == 0
    if isinstance(plugin, RandomVLDatasetPlugin):
        messages = list(plugin.build_messages())
        assert len(messages) == args.total_count
        texts = [message[0]['content'][0]['text'] for message in messages]
        assert all(message[0]['content'][1]['type'] == 'image_url' for message in messages)
    elif isinstance(plugin, RandomMultiTurnDatasetPlugin):
        conversations = list(plugin.build_messages())
        assert len(conversations) == max(args.parallel * 4, args.total_count + 1)
        assert all(1 <= len(conversation) <= 2 for conversation in conversations)
        texts = [turn.messages[0]['content'] for conversation in conversations for turn in conversation]
    else:
        conversation = plugin._trace_to_conversation(
            {
                'input_prompt_length': 8,
                'num_turns': 1,
                'assistant_response_length': [2],
                'tool_call_output_length': [5],
                'tool_call_latency': [0.1],
                'final_assistant_response_length': 3,
            }
        )
        assert [turn.max_tokens for turn in conversation] == [2, 3]
        texts = [turn.messages[0]['content'] for turn in conversation]
    assert all(plugin.tokenizer.encode(text, add_special_tokens=False)[:3] == expected for text in texts)


def test_real_bpe_boundary_can_change_raw_prefix(tmp_path: Path) -> None:
    backend = Tokenizer(models.BPE({'a': 0, 'b': 1, 'c': 2, 'ab': 3}, [('a', 'b')]))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, clean_up_tokenization_spaces=False)
    tokenizer.save_pretrained(tmp_path)
    plugin = RandomDatasetPlugin(_args(tokenizer_path=str(tmp_path), prefix_length=1, prefix_count=2))
    plugin.prefix_pool = [[0], [2]]
    # Same raw prefix, different body boundary: a+b merges to ab; a+c does not.
    assert plugin.generate_token_ids_only(1, 1, 0) == [0, 1]
    assert plugin.generate_token_ids_only(1, 2, 0) == [0, 2]
    first, _, _ = plugin.generate_token_sequence(1, 1, 0)
    second, _, _ = plugin.generate_token_sequence(1, 2, 0)
    assert tokenizer.encode(first, add_special_tokens=False)[0] == 3
    assert tokenizer.encode(second, add_special_tokens=False)[0] == 0
    assert plugin.prefix_pool == [[0], [2]]
