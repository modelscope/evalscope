"""Tests for unknown-key rejection and the eval_batch_size default resolution."""

import argparse
from typing import Optional

import pytest
from pydantic import ValidationError

from evalscope.arguments import add_argument
from evalscope.config import (
    DEFAULT_API_EVAL_BATCH_SIZE,
    DEFAULT_MODEL_ARGS_CHECKPOINT,
    REMOTE_API_EVAL_TYPES,
    SandboxTaskConfig,
    TaskConfig,
    parse_task_config,
)
from evalscope.perf.arguments import Arguments


def _cli_config(*argv: str) -> TaskConfig:
    parser = argparse.ArgumentParser()
    add_argument(parser)
    return parse_task_config(parser.parse_args(list(argv)))


def test_hyphenated_key_is_rejected_with_suggestion() -> None:
    with pytest.raises(ValueError) as excinfo:
        TaskConfig.from_dict({'model': 'x', 'api_url': 'http://localhost:8000/v1', 'eval-batch-size': 10})

    message = str(excinfo.value)
    assert 'eval-batch-size' in message
    assert "'eval-batch-size' -> 'eval_batch_size'" in message


def test_unknown_key_is_rejected_without_suggestion() -> None:
    with pytest.raises(ValueError, match='no_such_key_at_all'):
        TaskConfig.from_dict({'model': 'x', 'no_such_key_at_all': 1})


def test_unsuggestable_key_keeps_the_pydantic_error_type() -> None:
    with pytest.raises(ValidationError):
        TaskConfig.from_dict({'model': 'x', 'no_such_key_at_all': 1})


def test_nested_unknown_key_is_not_attributed_to_the_outer_model() -> None:
    nested = {'model': 'x', 'judge': {'models': [{'model_id': 'j', 'api_ur': 'oops'}]}}

    with pytest.raises(ValidationError) as excinfo:
        TaskConfig.from_dict(nested)

    message = str(excinfo.value)
    assert 'judge.models.0.api_ur' in message
    assert 'Did you mean' not in message


def test_perf_arguments_also_reject_unknown_keys() -> None:
    with pytest.raises(ValueError, match='parallel'):
        Arguments.from_dict({'model': 'x', 'url': 'http://localhost:8000/v1', 'parallell': 4})


def test_legacy_judge_keys_still_migrate() -> None:
    legacy = {'model': 'x', 'judge_strategy': 'auto', 'judge_model_args': {'model_id': 'judge-model'}}

    config = TaskConfig.from_dict(legacy)

    assert [model.model_id for model in config.judge.models] == ['judge-model']


@pytest.mark.parametrize(
    'api_url, explicit_batch_size, expected',
    [
        ('http://localhost:8000/v1', None, DEFAULT_API_EVAL_BATCH_SIZE),
        (None, None, 1),
        ('http://localhost:8000/v1', 1, 1),
        ('http://localhost:8000/v1', 10, 10),
    ],
)
def test_eval_batch_size_default_depends_on_eval_type(
    api_url: Optional[str], explicit_batch_size: Optional[int], expected: int
) -> None:
    overrides = {'model': 'x'}
    if api_url is not None:
        overrides['api_url'] = api_url
    if explicit_batch_size is not None:
        overrides['eval_batch_size'] = explicit_batch_size

    config = TaskConfig.from_dict(overrides)

    assert config.eval_batch_size == expected
    assert config.generation_config.batch_size == expected


@pytest.mark.parametrize('eval_type', sorted(REMOTE_API_EVAL_TYPES))
def test_every_remote_api_eval_type_gets_the_concurrent_default(eval_type: str) -> None:
    config = TaskConfig.from_dict({'model': 'x', 'eval_type': eval_type})

    assert config.eval_batch_size == DEFAULT_API_EVAL_BATCH_SIZE


@pytest.mark.parametrize('eval_type', ['llm_ckpt', 'mock_llm', 'text2image'])
def test_local_eval_types_keep_the_serial_default(eval_type: str) -> None:
    config = TaskConfig.from_dict({'model': 'x', 'eval_type': eval_type})

    assert config.eval_batch_size == 1


@pytest.mark.parametrize('alias, canonical', [('checkpoint', 'llm_ckpt'), ('server', 'openai_api')])
def test_deprecated_eval_type_alias_is_normalized(alias: str, canonical: str) -> None:
    config = TaskConfig.from_dict({'model': 'x', 'eval_type': alias})

    assert config.eval_type == canonical


def test_server_alias_inherits_the_remote_api_concurrent_default() -> None:
    config = TaskConfig.from_dict({'model': 'x', 'eval_type': 'server'})

    assert config.eval_batch_size == DEFAULT_API_EVAL_BATCH_SIZE


def test_checkpoint_alias_inherits_the_checkpoint_default_model_args() -> None:
    config = TaskConfig.from_dict({'model': '/path/to/model', 'eval_type': 'checkpoint'})

    assert config.model_args == DEFAULT_MODEL_ARGS_CHECKPOINT


def test_cli_omitting_the_flag_keeps_the_api_default() -> None:
    config = _cli_config('--model', 'x', '--api-url', 'http://localhost:8000/v1', '--datasets', 'gsm8k')

    assert config.eval_batch_size == DEFAULT_API_EVAL_BATCH_SIZE


def test_cli_explicit_flag_wins_over_the_api_default() -> None:
    config = _cli_config(
        '--model', 'x', '--api-url', 'http://localhost:8000/v1', '--datasets', 'gsm8k', '--eval-batch-size', '3'
    )

    assert config.eval_batch_size == 3


def test_legacy_sandbox_fields_fold_into_nested_sandbox() -> None:
    config = TaskConfig.from_dict({'model': 'x', 'sandbox_type': 'volcengine'})

    assert config.sandbox is not None
    assert config.sandbox.engine == 'volcengine'


def test_nested_sandbox_wins_over_legacy_fields() -> None:
    config = TaskConfig.from_dict({
        'model': 'x',
        'sandbox_type': 'volcengine',
        'sandbox': {
            'enabled': False,
            'engine': 'docker'
        },
    })

    assert config.sandbox.engine == 'docker'


def test_absent_sandbox_defaults_to_disabled() -> None:
    config = TaskConfig.from_dict({'model': 'x'})

    assert config.sandbox is not None
    assert config.sandbox.enabled is False


@pytest.mark.parametrize('falsy', ['false', '0', 0, False])
def test_falsy_use_sandbox_string_does_not_enable_sandbox(falsy: object) -> None:
    config = TaskConfig.from_dict({'model': 'x', 'use_sandbox': falsy})

    assert config.sandbox.enabled is False


def test_agent_config_is_set_only_on_both_serialization_paths() -> None:
    config = TaskConfig.from_dict({'model': 'x', 'api_url': 'u', 'agent_config': {'mode': 'native'}})

    assert config.to_dict()['agent_config'] == {'mode': 'native'}
    assert config._to_update_dict()['agent_config'] == {'mode': 'native'}


def test_to_dict_round_trip_is_stable() -> None:
    config = TaskConfig.from_dict({
        'model': 'x',
        'api_url': 'u',
        'generation_config': {
            'temperature': 0.5
        },
        'agent_config': {
            'mode': 'native'
        },
    })

    dumped = config.to_dict()
    redumped = TaskConfig.from_dict(dumped).to_dict()

    assert dumped == redumped


def test_serialization_paths_render_special_fields_per_purpose() -> None:
    config = TaskConfig.from_dict({'model': 'x', 'api_url': 'u', 'generation_config': {'temperature': 0.5}})

    yaml_dict = config.to_dict()
    update_dict = config._to_update_dict()

    assert isinstance(yaml_dict['sandbox'], dict)
    assert isinstance(update_dict['sandbox'], SandboxTaskConfig)
    assert yaml_dict['generation_config'] == update_dict['generation_config']


def test_update_merges_generation_config_and_recoerces() -> None:
    config = TaskConfig.from_dict({'model': 'x', 'api_url': 'u', 'generation_config': {'temperature': 0.5}})

    config.update({'generation_config': {'top_p': 0.9}, 'sandbox': {'engine': 'volcengine'}})

    merged = config.generation_config.model_dump(exclude_unset=True)
    assert merged['temperature'] == 0.5
    assert merged['top_p'] == 0.9
    assert isinstance(config.sandbox, SandboxTaskConfig)
    assert config.sandbox.engine == 'volcengine'
