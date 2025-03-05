import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evalscope.constants import DEFAULT_WORK_DIR


@dataclass
class Arguments:
    # Model and API
    model: str  # Model name or path
    model_id: Optional[str] = None  # Model identifier
    attn_implementation: Optional[str] = None  # Attention implementaion, only for local inference
    api: str = 'openai'  # API to be used (default: 'openai')
    tokenizer_path: Optional[str] = None  # Path to the tokenizer
    port: int = 8877  # Port number for the local API server

    # Connection settings
    url: str = 'http://127.0.0.1:8877/v1/chat/completions'  # URL for the API connection
    headers: Dict[str, Any] = field(default_factory=dict)  # Custom headers
    connect_timeout: int = 120  # Connection timeout in seconds
    read_timeout: int = 120  # Read timeout in seconds
    api_key: str = 'EMPTY'

    # Performance and parallelism
    number: Optional[int] = None  # Number of requests to be made
    parallel: int = 1  # Number of parallel requests
    rate: int = -1  # Rate limit for requests (default: -1, no limit)

    # Logging and debugging
    log_every_n_query: int = 10  # Log every N queries
    debug: bool = False  # Debug mode
    wandb_api_key: Optional[str] = None  # WandB API key for logging
    name: Optional[str] = None  # Name for the run

    # Output settings
    outputs_dir: str = DEFAULT_WORK_DIR

    # Prompt settings
    max_prompt_length: int = sys.maxsize  # Maximum length of the prompt
    min_prompt_length: int = 0  # Minimum length of the prompt
    prompt: Optional[str] = None  # The prompt text
    query_template: Optional[str] = None  # Template for the query

    # Dataset settings
    dataset: str = 'openqa'  # Dataset type (default: 'line_by_line')
    dataset_path: Optional[str] = None  # Path to the dataset

    # Response settings
    frequency_penalty: Optional[float] = None  # Frequency penalty for the response
    logprobs: Optional[bool] = None  # Whether to log probabilities
    max_tokens: Optional[int] = 2048  # Maximum number of tokens in the response
    min_tokens: Optional[int] = None  # Minimum number of tokens in the response
    n_choices: Optional[int] = None  # Number of response choices
    seed: Optional[int] = 42  # Random seed for reproducibility
    stop: Optional[List[str]] = field(default_factory=list)  # Stop sequences for the response
    stop_token_ids: Optional[List[str]] = field(default_factory=list)  # Stop token IDs for the response
    stream: Optional[bool] = None  # Whether to stream the response
    temperature: Optional[float] = None  # Temperature setting for the response
    top_p: Optional[float] = None  # Top-p (nucleus) sampling setting for the response
    top_k: Optional[int] = None  # Top-k sampling setting for the response

    @staticmethod
    def from_args(args):
        return Arguments(
            model=args.model,
            attn_implementation=args.attn_implementation,
            url=args.url,
            port=args.port,
            api_key=args.api_key,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            number=args.number,
            parallel=args.parallel,
            rate=args.rate,
            log_every_n_query=args.log_every_n_query,
            headers=args.headers,
            wandb_api_key=args.wandb_api_key,
            name=args.name,
            outputs_dir=args.outputs_dir,
            debug=args.debug,
            tokenizer_path=args.tokenizer_path,
            api=args.api,
            max_prompt_length=args.max_prompt_length,
            min_prompt_length=args.min_prompt_length,
            prompt=args.prompt,
            query_template=args.query_template,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            frequency_penalty=args.frequency_penalty,
            logprobs=args.logprobs,
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
            n_choices=args.n_choices,
            seed=args.seed,
            stop=args.stop,
            stop_token_ids=args.stop_token_ids,
            stream=args.stream,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    def __post_init__(self):
        self.headers = self.headers or {}  # Default to empty dictionary
        if self.api_key:
            # Assuming the API key is used as a Bearer token
            self.headers['Authorization'] = f'{self.api_key}'
        self.model_id = os.path.basename(self.model)

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4, default=str, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class ParseKVAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if not values:
            setattr(namespace, self.dest, {})
        else:
            try:
                kv_dict = dict(kv.split('=') for kv in values)
                setattr(namespace, self.dest, kv_dict)
            except ValueError as e:
                parser.error(f'Error parsing key-value pairs: {e}')


def add_argument(parser: argparse.ArgumentParser):
    # yapf: disable
    # Model and API
    parser.add_argument('--model', type=str, required=True, help='The test model name.')
    parser.add_argument('--attn-implementation', required=False, default=None, help='Attention implementaion')
    parser.add_argument('--api', type=str, default='openai', help='Specify the service API')
    parser.add_argument(
        '--tokenizer-path', type=str, required=False, default=None, help='Specify the tokenizer weight path')

    # Connection settings
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8877/v1/chat/completions')
    parser.add_argument('--port', type=int, default=8877, help='The port for local inference')
    parser.add_argument('--headers', nargs='+', dest='headers', action=ParseKVAction, help='Extra HTTP headers')
    parser.add_argument('--api-key', type=str, required=False, default='EMPTY', help='The API key for authentication')
    parser.add_argument('--connect-timeout', type=int, default=120, help='The network connection timeout')
    parser.add_argument('--read-timeout', type=int, default=120, help='The network read timeout')

    # Performance and parallelism
    parser.add_argument('-n', '--number', type=int, default=None, help='How many requests to be made')
    parser.add_argument('--parallel', type=int, default=1, help='Set number of concurrency requests, default 1')
    parser.add_argument('--rate', type=int, default=-1, help='Number of requests per second. default None')

    # Logging and debugging
    parser.add_argument('--log-every-n-query', type=int, default=10, help='Logging every n query')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug request send')
    parser.add_argument('--wandb-api-key', type=str, default=None, help='The wandb API key')
    parser.add_argument('--name', type=str, help='The wandb db result name and result db name')

    # Prompt settings
    parser.add_argument('--max-prompt-length', type=int, default=sys.maxsize, help='Maximum input prompt length')
    parser.add_argument('--min-prompt-length', type=int, default=0, help='Minimum input prompt length')
    parser.add_argument('--prompt', type=str, required=False, default=None, help='Specified the request prompt')
    parser.add_argument('--query-template', type=str, default=None, help='Specify the query template')

    # Output settings
    parser.add_argument('--outputs-dir', help='Outputs dir.', default='outputs')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='openqa', help='Specify the dataset')
    parser.add_argument('--dataset-path', type=str, required=False, help='Path to the dataset file')

    # Response settings
    parser.add_argument('--frequency-penalty', type=float, help='The frequency_penalty value', default=None)
    parser.add_argument('--logprobs', action='store_true', help='The logprobs', default=None)
    parser.add_argument(
        '--max-tokens', type=int, help='The maximum number of tokens that can be generated', default=2048)
    parser.add_argument(
        '--min-tokens', type=int, help='The minimum number of tokens that can be generated', default=None)
    parser.add_argument('--n-choices', type=int, help='How many completion choices to generate', default=None)
    parser.add_argument('--seed', type=int, help='The random seed', default=42)
    parser.add_argument('--stop', nargs='*', help='The stop tokens', default=None)
    parser.add_argument('--stop-token-ids', nargs='*', help='Set the stop token IDs', default=None)
    parser.add_argument('--stream', action='store_true', help='Stream output with SSE', default=None)
    parser.add_argument('--temperature', type=float, help='The sample temperature', default=None)
    parser.add_argument('--top-p', type=float, help='Sampling top p', default=None)
    parser.add_argument('--top-k', type=int, help='Sampling top k', default=None)

    # yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark LLM service performance.')
    add_argument(parser)
    return parser.parse_args()
