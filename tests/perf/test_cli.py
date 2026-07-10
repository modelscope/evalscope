from argparse import ArgumentParser

from evalscope.perf.config.cli import add_cli_arguments, config_from_namespace
from evalscope.perf.config.models import ClosedLoopLoad, OpenLoopLoad


def test_cli_builds_typed_open_loop_config() -> None:
    parser = ArgumentParser()
    add_cli_arguments(parser)
    args = parser.parse_args([
        '--model',
        'fake',
        '--mode',
        'open_loop',
        '--request-rate',
        '5',
        '--requests',
        '20',
        '--max-outstanding',
        '3',
    ])
    config = config_from_namespace(args)
    load = config.suite.loads[0]
    assert isinstance(load, OpenLoopLoad)
    assert load.request_rate == 5
    assert load.max_outstanding == 3


def test_cli_repeatable_load_json() -> None:
    parser = ArgumentParser()
    add_cli_arguments(parser)
    args = parser.parse_args([
        '--model',
        'fake',
        '--load',
        '{"mode":"closed_loop","concurrency":1,"request_count":2}',
        '--load',
        '{"mode":"closed_loop","concurrency":2,"request_count":4}',
    ])
    assert len(config_from_namespace(args).suite.loads) == 2


def test_cli_parses_sla_config() -> None:
    parser = ArgumentParser()
    add_cli_arguments(parser)
    args = parser.parse_args([
        '--model',
        'fake',
        '--sla-config',
        '{"variable":"concurrency","criteria":[{"metric":"p99_latency","op":"<","value":"2"}]}',
    ])
    config = config_from_namespace(args)
    assert config.sla is not None
    assert config.sla.variable == 'concurrency'


def test_legacy_cli_sweep_translates_to_explicit_loads(tmp_path) -> None:
    dataset = tmp_path / 'prompts.txt'
    dataset.write_text('hello\n', encoding='utf-8')
    parser = ArgumentParser()
    add_cli_arguments(parser)
    args = parser.parse_args([
        '--model',
        'fake',
        '--api',
        'openai',
        '--url',
        'http://localhost:8000/v1/chat/completions',
        '--dataset',
        'custom',
        '--dataset-path',
        str(dataset),
        '--number',
        '10',
        '20',
        '--parallel',
        '1',
        '2',
        '--warmup-num',
        '0.1',
        '--outputs-dir',
        str(tmp_path),
        '--no-test-connection',
        '--max-tokens',
        '16',
        '32',
    ])
    config = config_from_namespace(args)
    assert config.target.protocol == 'openai_chat'
    assert config.target.skip_connection_test
    assert config.workload.name == 'custom'
    assert config.generation.max_tokens == (16, 32)
    assert [load.concurrency for load in config.suite.loads if isinstance(load, ClosedLoopLoad)] == [1, 2]
    assert [load.request_count for load in config.suite.loads] == [10, 20]
    assert config.suite.loads[0].warmup.ratio == 0.1


def test_legacy_open_loop_and_tokenized_prompt_translation() -> None:
    parser = ArgumentParser()
    add_cli_arguments(parser)
    args = parser.parse_args([
        '--model',
        'fake',
        '--open-loop',
        '--number',
        '10',
        '20',
        '--rate',
        '1',
        '2',
        '--tokenize-prompt',
        '--tokenizer-path',
        'fake-tokenizer',
        '--dataset',
        'random',
    ])
    config = config_from_namespace(args)
    assert config.target.protocol == 'openai_completions'
    assert [load.request_rate for load in config.suite.loads if isinstance(load, OpenLoopLoad)] == [1, 2]
    assert config.workload.options['tokenize_prompt'] is True
