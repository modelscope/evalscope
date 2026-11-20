"""Tests for the ``key=value,key=value`` form of ``--model-args`` and ``--generation-config``."""

import argparse
from typing import Any, Dict

import pytest

from evalscope.arguments import ParseStrArgsAction, add_argument
from evalscope.config import parse_task_config


def _parse(value: str) -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-args', action=ParseStrArgsAction)
    return parser.parse_args(['--model-args', value]).model_args


def test_plain_pairs_are_unchanged() -> None:
    assert _parse('revision=master,precision=torch.float16,device_map=auto') == {
        'revision': 'master',
        'precision': 'torch.float16',
        'device_map': 'auto',
    }
    assert _parse('do_sample=true,temperature=0.5,top_k=-1,stream=False') == {
        'do_sample': True,
        'temperature': 0.5,
        'top_k': -1,
        'stream': False,
    }


def test_json_is_still_read_first() -> None:
    assert _parse('{"stop_seqs": ["a", "b"], "temperature": 0}') == {'stop_seqs': ['a', 'b'], 'temperature': 0}


def test_commas_inside_a_list_stay_in_the_value() -> None:
    assert _parse("max_tokens=512,stop_seqs=['</s>','<|im_end|>'],temperature=0") == {
        'max_tokens': 512,
        'stop_seqs': ['</s>', '<|im_end|>'],
        'temperature': 0,
    }


def test_commas_inside_a_dict_stay_in_the_value() -> None:
    assert _parse("max_memory={0: '10GiB', 1: '20GiB'},dtype=auto") == {
        'max_memory': {0: '10GiB', 1: '20GiB'},
        'dtype': 'auto',
    }


def test_brackets_and_commas_inside_quoted_items_do_not_end_the_value() -> None:
    assert _parse("stop_seqs=['a,b','c]d'],seed=1") == {'stop_seqs': ['a,b', 'c]d'], 'seed': 1}


def test_apostrophe_in_a_plain_value_requires_quotes(capsys: pytest.CaptureFixture) -> None:
    assert _parse('name="it\'s",top_p=0.9') == {'name': "it's", 'top_p': 0.9}

    with pytest.raises(SystemExit) as excinfo:
        _parse("name=it's,top_p=0.9")

    assert excinfo.value.code == 2
    assert 'expected comma-separated key=value pairs' in capsys.readouterr().err


def test_escaped_backslash_before_a_closing_quote_is_handled_by_python() -> None:
    value = r"stop_seqs=['C:\\'],max_memory={0:'1GiB',1:'2GiB'}"

    assert _parse(value) == {
        'stop_seqs': ['C:\\'],
        'max_memory': {0: '1GiB', 1: '2GiB'},
    }


def test_pair_without_equals_is_a_usage_error(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse('temperature=0.5,stop_seqs')

    assert excinfo.value.code == 2
    assert 'expected comma-separated key=value pairs' in capsys.readouterr().err


def test_unclosed_literal_is_a_usage_error(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse("stop_seqs=['a','b'")

    assert excinfo.value.code == 2
    assert 'expected comma-separated key=value pairs' in capsys.readouterr().err


def test_stop_sequences_reach_the_task_config() -> None:
    parser = argparse.ArgumentParser()
    add_argument(parser)
    config = parse_task_config(
        parser.parse_args([
            '--model',
            'parse-str-args-test',
            '--eval-type',
            'mock_llm',
            '--datasets',
            'general_mcq',
            '--generation-config',
            "max_tokens=64,stop_seqs=['Question:','Answer:']",
        ])
    )

    assert config.generation_config.max_tokens == 64
    assert config.generation_config.stop_seqs == ['Question:', 'Answer:']
