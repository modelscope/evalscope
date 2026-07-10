from __future__ import annotations

import argparse
import json
from pydantic import TypeAdapter
from typing import Any, Dict, List

from evalscope.perf.config.models import (
    BenchmarkSuite,
    ClosedLoopLoad,
    ConversationLoad,
    GenerationConfig,
    LoadSpec,
    OpenLoopLoad,
    OutputConfig,
    PerfConfig,
    RuntimeConfig,
    SLAConfig,
    TargetConfig,
    WarmupConfig,
    WorkloadConfig,
)
from evalscope.utils.logger import get_logger

logger = get_logger()

_LEGACY_API_MAP = {
    'openai': 'openai_chat',
    'default': 'openai_chat',
    'dashscope': 'openai_chat',
    'anthropic': 'openai_chat',
    'gemini': 'openai_chat',
    'custom': 'openai_chat',
    'openai_responses': 'openai_responses',
    'openai_response': 'openai_responses',
    'responses': 'openai_responses',
    'openai_embedding': 'openai_embedding',
    'embedding': 'openai_embedding',
    'openai_rerank': 'openai_rerank',
    'rerank': 'openai_rerank',
    'local': 'openai_chat',
    'local_vllm': 'openai_chat',
}


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--model', required=True)
    parser.add_argument(
        '--protocol',
        choices=['openai_chat', 'openai_completions', 'openai_responses', 'openai_embedding', 'openai_rerank'],
        default='openai_chat',
    )
    parser.add_argument('--target-kind', choices=['remote', 'local_transformers', 'local_vllm'], default='remote')
    parser.add_argument('--base-url', default='http://127.0.0.1:8877/v1')
    parser.add_argument('--api-key')
    parser.add_argument('--header', action='append', default=[], help='HTTP header in KEY=VALUE form')
    parser.add_argument('--tokenizer')
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--attn-implementation')
    parser.add_argument('--connect-timeout', type=float)
    parser.add_argument('--read-timeout', type=float)
    parser.add_argument('--total-timeout', type=float, default=21600)
    parser.add_argument('--skip-connection-test', action='store_true')

    parser.add_argument('--workload', default='openqa')
    parser.add_argument('--workload-path')
    parser.add_argument('--data-source', choices=['modelscope', 'huggingface', 'local'], default='modelscope')
    parser.add_argument('--prompt')
    parser.add_argument('--workload-options', type=json.loads, default={})
    parser.add_argument('--min-prompt-length', type=int, default=0)
    parser.add_argument('--max-prompt-length', type=int, default=131072)

    parser.add_argument('--max-tokens', type=int, nargs='+', default=[2048])
    parser.add_argument('--min-tokens', type=int)
    parser.add_argument('--frequency-penalty', type=float)
    parser.add_argument('--repetition-penalty', type=float)
    parser.add_argument('--logprobs', action='store_true', default=None)
    parser.add_argument('--n-choices', type=int)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float)
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--stop', nargs='*')
    parser.add_argument('--stop-token-ids', nargs='*', type=int)
    parser.add_argument('--no-stream', action='store_true')
    parser.add_argument('--generation-extra', type=json.loads, default={})

    parser.add_argument('--load', action='append', type=json.loads, help='Repeatable JSON load specification')
    parser.add_argument('--mode', choices=['closed_loop', 'open_loop', 'conversation'], default='closed_loop')
    parser.add_argument('--requests', type=int, default=1000)
    parser.add_argument('--conversations', type=int)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--request-rate', type=float)
    parser.add_argument('--max-outstanding', type=int, default=512)
    parser.add_argument('--overflow-policy', choices=['record_drop', 'fail'], default='record_drop')
    parser.add_argument('--arrival', choices=['poisson', 'calibrated_poisson', 'constant'], default='poisson')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--warmup-count', type=int)
    parser.add_argument('--warmup-ratio', type=float)
    parser.add_argument('--sleep-between-runs', type=float, default=0)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--dataset-workers', type=int, default=0)
    parser.add_argument('--queue-size', type=int, default=1024)
    parser.add_argument('--db-commit-interval', type=int, default=1000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--visualizer', choices=['wandb', 'swanlab', 'clearml'])
    parser.add_argument('--visualizer-project', default='evalscope-perf')
    parser.add_argument('--visualizer-name')
    parser.add_argument('--output-root', default='outputs/perf')
    parser.add_argument('--run-id')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-html-report', action='store_true')
    parser.add_argument('--no-console-report', action='store_true')
    parser.add_argument('--sla-config', type=json.loads, help='SLA search configuration as JSON')
    _add_legacy_arguments(parser)


def config_from_namespace(args: argparse.Namespace) -> PerfConfig:
    legacy = _apply_legacy_arguments(args)
    headers = _parse_headers(args.header)
    headers.update(_parse_headers(args.legacy_headers or []))
    warmup = _warmup(args)
    if args.load:
        if legacy['load']:
            raise ValueError('Do not combine legacy load arguments with --load')
        adapter = TypeAdapter(LoadSpec)
        loads = [adapter.validate_python(item) for item in args.load]
    elif legacy['load']:
        loads = _legacy_loads(args, warmup)
    else:
        loads = [_single_load(args, warmup)]
    return PerfConfig(
        target=TargetConfig(
            model=args.model,
            protocol=args.protocol,
            kind=args.target_kind,
            base_url=args.base_url,
            api_key=args.api_key,
            headers=headers,
            tokenizer=args.tokenizer,
            port=args.port,
            attn_implementation=args.attn_implementation,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            total_timeout=args.total_timeout,
            skip_connection_test=args.skip_connection_test,
        ),
        workload=WorkloadConfig(
            name='prompt' if args.prompt and args.workload == 'openqa' else args.workload,
            path=args.workload_path,
            data_source=args.data_source,
            prompt=args.prompt,
            options=args.workload_options,
            min_prompt_length=args.min_prompt_length,
            max_prompt_length=args.max_prompt_length,
        ),
        generation=GenerationConfig(
            max_tokens=_max_tokens(args.max_tokens),
            min_tokens=args.min_tokens,
            frequency_penalty=args.frequency_penalty,
            repetition_penalty=args.repetition_penalty,
            logprobs=args.logprobs,
            n_choices=args.n_choices,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop=args.stop,
            stop_token_ids=args.stop_token_ids,
            stream=not args.no_stream,
            extra=args.generation_extra,
        ),
        suite=BenchmarkSuite(loads=loads, sleep_between_runs=args.sleep_between_runs),
        runtime=RuntimeConfig(
            seed=args.seed,
            dataset_workers=args.dataset_workers,
            queue_size=args.queue_size,
            db_commit_interval=args.db_commit_interval,
            debug=args.debug,
            progress=args.progress,
            visualizer=args.visualizer,
            visualizer_project=args.visualizer_project,
            visualizer_name=args.visualizer_name,
        ),
        output=OutputConfig(
            root=args.output_root,
            run_id=args.run_id,
            overwrite=args.overwrite,
            html_report=not args.no_html_report,
            console_report=not args.no_console_report,
        ),
        sla=SLAConfig.model_validate(args.sla_config) if args.sla_config else None,
    )


def _add_legacy_arguments(parser: argparse.ArgumentParser) -> None:
    hidden = argparse.SUPPRESS
    parser.add_argument('--api', dest='legacy_api', help=hidden)
    parser.add_argument('--url', dest='legacy_url', help=hidden)
    parser.add_argument('--tokenizer-path', dest='legacy_tokenizer', help=hidden)
    parser.add_argument('--headers', dest='legacy_headers', nargs='+', help=hidden)
    parser.add_argument('--no-test-connection', dest='legacy_skip_connection', action='store_true', help=hidden)
    parser.add_argument('-n', '--number', dest='legacy_number', type=int, nargs='+', help=hidden)
    parser.add_argument('--parallel', dest='legacy_parallel', type=int, nargs='+', help=hidden)
    parser.add_argument('--rate', dest='legacy_rate', type=float, nargs='+', help=hidden)
    parser.add_argument('--warmup-num', dest='legacy_warmup', type=float, help=hidden)
    parser.add_argument('--open-loop', dest='legacy_open_loop', action='store_true', help=hidden)
    parser.add_argument('--sleep-interval', dest='legacy_sleep', type=float, help=hidden)
    parser.add_argument('--num-workers', dest='legacy_workers', type=int, help=hidden)
    parser.add_argument('--log-every-n-query', dest='legacy_log_every', type=int, help=hidden)
    parser.add_argument('--enable-progress-tracker', dest='legacy_progress', action='store_true', help=hidden)
    parser.add_argument('--name', dest='legacy_name', help=hidden)
    parser.add_argument('--outputs-dir', dest='legacy_output_root', help=hidden)
    parser.add_argument('--no-timestamp', dest='legacy_no_timestamp', action='store_true', help=hidden)
    parser.add_argument('--dataset', dest='legacy_workload', help=hidden)
    parser.add_argument('--dataset-path', dest='legacy_workload_path', help=hidden)
    parser.add_argument('--dataset-offset', dest='legacy_dataset_offset', type=int, help=hidden)
    parser.add_argument('--prefix-length', dest='legacy_prefix_length', type=int, help=hidden)
    parser.add_argument('--query-template', dest='legacy_query_template', help=hidden)
    parser.add_argument(
        '--apply-chat-template', dest='legacy_apply_chat_template', action=argparse.BooleanOptionalAction, help=hidden
    )
    parser.add_argument('--image-width', dest='legacy_image_width', type=int, help=hidden)
    parser.add_argument('--image-height', dest='legacy_image_height', type=int, help=hidden)
    parser.add_argument('--image-format', dest='legacy_image_format', help=hidden)
    parser.add_argument('--image-num', dest='legacy_image_num', type=int, help=hidden)
    parser.add_argument('--image-patch-size', dest='legacy_image_patch_size', type=int, help=hidden)
    parser.add_argument('--extra-args', dest='legacy_generation_extra', type=json.loads, help=hidden)
    parser.add_argument('--stream', dest='legacy_stream', action='store_true', default=None, help=hidden)
    parser.add_argument('--tokenize-prompt', dest='legacy_tokenize_prompt', action='store_true', help=hidden)
    parser.add_argument('--multi-turn', dest='legacy_multi_turn', action='store_true', help=hidden)
    parser.add_argument('--min-turns', dest='legacy_min_turns', type=int, help=hidden)
    parser.add_argument('--max-turns', dest='legacy_max_turns', type=int, help=hidden)
    parser.add_argument('--multi-turn-args', dest='legacy_multi_turn_args', type=json.loads, help=hidden)
    parser.add_argument('--queue-size-multiplier', dest='legacy_queue_multiplier', type=int, help=hidden)
    parser.add_argument('--in-flight-task-multiplier', dest='legacy_task_multiplier', type=int, help=hidden)
    parser.add_argument('--wandb-api-key', dest='legacy_wandb_key', help=hidden)
    parser.add_argument('--swanlab-api-key', dest='legacy_swanlab_key', help=hidden)
    parser.add_argument('--swanlab-host', dest='legacy_swanlab_host', help=hidden)
    parser.add_argument('--sla-auto-tune', dest='legacy_sla', action='store_true', help=hidden)
    parser.add_argument('--sla-variable', dest='legacy_sla_variable', choices=['parallel', 'rate'], help=hidden)
    parser.add_argument('--sla-params', dest='legacy_sla_params', type=json.loads, help=hidden)
    parser.add_argument('--sla-num-runs', dest='legacy_sla_runs', type=int, help=hidden)
    parser.add_argument('--sla-upper-bound', dest='legacy_sla_upper', type=int, help=hidden)
    parser.add_argument('--sla-lower-bound', dest='legacy_sla_lower', type=int, help=hidden)
    parser.add_argument('--sla-fixed-parallel', dest='legacy_sla_fixed_parallel', type=int, help=hidden)
    parser.add_argument('--sla-number-multiplier', dest='legacy_sla_number_multiplier', type=float, help=hidden)


def _apply_legacy_arguments(args: argparse.Namespace) -> Dict[str, bool]:
    mappings = []

    def use(attribute: str, old: str, new: str) -> bool:
        value = getattr(args, attribute)
        if value is not None and value is not False:
            mappings.append(f'{old} -> {new}')
            return True
        return False

    if use('legacy_api', '--api', '--protocol/--target-kind'):
        if args.legacy_api not in _LEGACY_API_MAP:
            raise ValueError(f'Unsupported legacy --api value: {args.legacy_api}')
        args.protocol = _LEGACY_API_MAP[args.legacy_api]
        if args.legacy_api == 'local':
            args.target_kind = 'local_transformers'
        elif args.legacy_api == 'local_vllm':
            args.target_kind = 'local_vllm'
    if use('legacy_url', '--url', '--base-url'):
        args.base_url = args.legacy_url
    if use('legacy_tokenizer', '--tokenizer-path', '--tokenizer'):
        args.tokenizer = args.legacy_tokenizer
    if use('legacy_skip_connection', '--no-test-connection', '--skip-connection-test'):
        args.skip_connection_test = True
    if use('legacy_workload', '--dataset', '--workload'):
        args.workload = args.legacy_workload
    if use('legacy_workload_path', '--dataset-path', '--workload-path'):
        args.workload_path = args.legacy_workload_path
    if use('legacy_output_root', '--outputs-dir', '--output-root'):
        args.output_root = args.legacy_output_root
    if use('legacy_sleep', '--sleep-interval', '--sleep-between-runs'):
        args.sleep_between_runs = args.legacy_sleep
    if use('legacy_workers', '--num-workers', '--dataset-workers'):
        args.dataset_workers = args.legacy_workers
    if use('legacy_progress', '--enable-progress-tracker', '--progress'):
        args.progress = True
    if use('legacy_name', '--name', '--run-id/--visualizer-name'):
        args.visualizer_name = args.legacy_name
    if use('legacy_no_timestamp', '--no-timestamp', '--run-id'):
        args.run_id = args.legacy_name or 'perf'
    if use('legacy_generation_extra', '--extra-args', '--generation-extra'):
        args.generation_extra = args.legacy_generation_extra
    if use('legacy_stream', '--stream', 'streaming is enabled by default'):
        args.no_stream = not args.legacy_stream
    if use('legacy_tokenize_prompt', '--tokenize-prompt', '--protocol openai_completions'):
        args.protocol = 'openai_completions'

    options = dict(args.workload_options)
    option_fields = {
        'legacy_dataset_offset': ('dataset_offset', '--dataset-offset'),
        'legacy_prefix_length': ('prefix_length', '--prefix-length'),
        'legacy_query_template': ('query_template', '--query-template'),
        'legacy_apply_chat_template': ('apply_chat_template', '--apply-chat-template'),
        'legacy_image_width': ('image_width', '--image-width'),
        'legacy_image_height': ('image_height', '--image-height'),
        'legacy_image_format': ('image_format', '--image-format'),
        'legacy_image_num': ('image_num', '--image-num'),
        'legacy_image_patch_size': ('image_patch_size', '--image-patch-size'),
        'legacy_min_turns': ('min_turns', '--min-turns'),
        'legacy_max_turns': ('max_turns', '--max-turns'),
        'legacy_multi_turn_args': ('multi_turn_args', '--multi-turn-args'),
    }
    for attribute, (key, old) in option_fields.items():
        selected = (
            getattr(args, attribute) is not None
            if attribute == 'legacy_apply_chat_template' else use(attribute, old, f'--workload-options {key}')
        )
        if selected:
            if attribute == 'legacy_apply_chat_template':
                mappings.append(f'{old} -> --workload-options {key}')
            options[key] = getattr(args, attribute)
    if args.legacy_tokenize_prompt:
        options['tokenize_prompt'] = True
    args.workload_options = options

    legacy_load_flags = [
        use(attribute, old, new) for attribute, old, new in (
            ('legacy_number', '--number/-n', '--requests/--conversations'),
            ('legacy_parallel', '--parallel', '--concurrency'),
            ('legacy_rate', '--rate', '--request-rate'),
            ('legacy_warmup', '--warmup-num', '--warmup-count/--warmup-ratio'),
            ('legacy_open_loop', '--open-loop', '--mode open_loop'),
            ('legacy_multi_turn', '--multi-turn', '--mode conversation'),
        )
    ]
    legacy_load = any(legacy_load_flags)
    ignored = (
        ('legacy_queue_multiplier', '--queue-size-multiplier'),
        ('legacy_task_multiplier', '--in-flight-task-multiplier'),
        ('legacy_wandb_key', '--wandb-api-key'),
        ('legacy_swanlab_key', '--swanlab-api-key'),
        ('legacy_swanlab_host', '--swanlab-host'),
        ('legacy_sla_fixed_parallel', '--sla-fixed-parallel'),
        ('legacy_sla_number_multiplier', '--sla-number-multiplier'),
        ('legacy_log_every', '--log-every-n-query'),
    )
    for attribute, old in ignored:
        use(attribute, old, 'no direct equivalent; ignored')
    _legacy_sla(args, use)
    if mappings:
        logger.warning(
            'Deprecated perf CLI arguments are supported by a translation layer. '
            f'Please migrate to the new arguments: {", ".join(mappings)}'
        )
    return {'load': legacy_load}


def _legacy_sla(args: argparse.Namespace, use) -> None:
    used = [
        use(attribute, old, '--sla-config') for attribute, old in (
            ('legacy_sla', '--sla-auto-tune'),
            ('legacy_sla_variable', '--sla-variable'),
            ('legacy_sla_params', '--sla-params'),
            ('legacy_sla_runs', '--sla-num-runs'),
            ('legacy_sla_upper', '--sla-upper-bound'),
            ('legacy_sla_lower', '--sla-lower-bound'),
        )
    ]
    if not any(used):
        return
    criteria = args.legacy_sla_params
    if isinstance(criteria, dict):
        criteria = [criteria]
    args.sla_config = {
        'variable': 'request_rate' if args.legacy_sla_variable == 'rate' else 'concurrency',
        'criteria': criteria or [],
        'objective': None if criteria else 'max_rps',
        'lower_bound': args.legacy_sla_lower or 1,
        'upper_bound': args.legacy_sla_upper or 65536,
        'repetitions': args.legacy_sla_runs or 3,
    }


def _warmup(args: argparse.Namespace) -> WarmupConfig:
    if args.legacy_warmup is None:
        return WarmupConfig(count=args.warmup_count, ratio=args.warmup_ratio)
    if args.warmup_count is not None or args.warmup_ratio is not None:
        raise ValueError('Do not combine --warmup-num with new warmup arguments')
    if args.legacy_warmup >= 1:
        return WarmupConfig(count=int(args.legacy_warmup))
    if args.legacy_warmup > 0:
        return WarmupConfig(ratio=args.legacy_warmup)
    return WarmupConfig()


def _legacy_loads(args: argparse.Namespace, warmup: WarmupConfig) -> List[LoadSpec]:
    numbers = args.legacy_number or [args.requests]
    common = {'duration': args.duration, 'warmup': warmup}
    if args.legacy_multi_turn:
        concurrency = _broadcast(args.legacy_parallel or [args.concurrency], len(numbers), '--parallel')
        return [
            ConversationLoad(
                concurrency=value,
                conversation_count=number,
                max_turns=args.workload_options.get('max_turns'),
                **common,
            ) for number, value in zip(numbers, concurrency)
        ]
    if args.legacy_open_loop:
        rates = _broadcast(args.legacy_rate or [], len(numbers), '--rate')
        return [
            OpenLoopLoad(
                request_rate=rate,
                request_count=number,
                max_outstanding=args.max_outstanding,
                overflow_policy=args.overflow_policy,
                arrival=args.arrival,
                **common,
            ) for number, rate in zip(numbers, rates)
        ]
    concurrency = _broadcast(args.legacy_parallel or [args.concurrency], len(numbers), '--parallel')
    return [
        ClosedLoopLoad(concurrency=value, request_count=number, **common)
        for number, value in zip(numbers, concurrency)
    ]


def _broadcast(values: List[Any], size: int, name: str) -> List[Any]:
    if len(values) == size:
        return values
    if len(values) == 1:
        return values * size
    raise ValueError(f'{name} must contain one value or match the number of --number values')


def _max_tokens(values: List[int]):
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return tuple(values)
    raise ValueError('--max-tokens accepts one value or a min/max pair')


def _single_load(args: argparse.Namespace, warmup: WarmupConfig):
    common = {'duration': args.duration, 'warmup': warmup}
    if args.mode == 'closed_loop':
        return ClosedLoopLoad(concurrency=args.concurrency, request_count=args.requests, **common)
    if args.mode == 'open_loop':
        if args.request_rate is None:
            raise ValueError('--request-rate is required in open_loop mode')
        return OpenLoopLoad(
            request_rate=args.request_rate,
            request_count=args.requests,
            max_outstanding=args.max_outstanding,
            overflow_policy=args.overflow_policy,
            arrival=args.arrival,
            **common,
        )
    return ConversationLoad(
        concurrency=args.concurrency,
        conversation_count=args.conversations or args.requests,
        max_turns=args.workload_options.get('max_turns'),
        **common,
    )


def _parse_headers(values: List[str]) -> Dict[str, str]:
    headers = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'Invalid --header {value!r}; expected KEY=VALUE')
        key, item = value.split('=', 1)
        headers[key.strip()] = item.strip()
    return headers
