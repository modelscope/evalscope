import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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
    connect_timeout: int = 600  # Connection timeout in seconds
    read_timeout: int = 600  # Read timeout in seconds
    api_key: Optional[str] = None
    no_test_connection: bool = False  # Test the connection before starting the benchmark

    # Performance and parallelism
    number: Union[int, List[int]] = 1000  # Number of requests to be made
    parallel: Union[int, List[int]] = 1  # Number of parallel requests
    rate: int = -1  # Rate limit for requests (default: -1, no limit)

    # Logging and debugging
    log_every_n_query: int = 10  # Log every N queries
    debug: bool = False  # Debug mode
    wandb_api_key: Optional[str] = None  # WandB API key for logging
    swanlab_api_key: Optional[str] = None  # SwanLab API key for logging
    name: Optional[str] = None  # Name for the run

    # Output settings
    outputs_dir: str = DEFAULT_WORK_DIR

    # Prompt settings
    max_prompt_length: int = 131072  # Maximum length of the prompt
    min_prompt_length: int = 0  # Minimum length of the prompt
    prefix_length: int = 0  # Length of the prefix, only for random dataset
    prompt: Optional[str] = None  # The prompt text
    query_template: Optional[str] = None  # Template for the query
    apply_chat_template: Optional[bool] = None  # Whether to apply chat template

    # Dataset settings
    dataset: str = 'openqa'  # Dataset type (default: 'line_by_line')
    dataset_path: Optional[str] = None  # Path to the dataset

    # Response settings
    frequency_penalty: Optional[float] = None  # Frequency penalty for the response
    logprobs: Optional[bool] = None  # Whether to log probabilities
    max_tokens: Optional[int] = 2048  # Maximum number of tokens in the response
    min_tokens: Optional[int] = None  # Minimum number of tokens in the response
    n_choices: Optional[int] = None  # Number of response choices
    seed: Optional[int] = 0  # Random seed for reproducibility
    stop: Optional[List[str]] = None  # Stop sequences for the response
    stop_token_ids: Optional[List[str]] = None  # Stop token IDs for the response
    stream: Optional[bool] = True  # Whether to stream the response
    temperature: float = 0.0  # Temperature setting for the response
    top_p: Optional[float] = None  # Top-p (nucleus) sampling setting for the response
    top_k: Optional[int] = None  # Top-k sampling setting for the response
    extra_args: Optional[Dict[str, Any]] = None  # Extra arguments

    @staticmethod
    def from_args(args):
        # Convert Namespace to a dictionary and filter out None values
        args_dict = {k: v for k, v in vars(args).items() if v is not None}

        if 'func' in args_dict:
            del args_dict['func']  # Note: compat CLI arguments
        return Arguments(**args_dict)

    def __post_init__(self):
        # Set the default headers
        self.headers = self.headers or {}  # Default to empty dictionary
        if self.api_key:
            # Assuming the API key is used as a Bearer token
            self.headers['Authorization'] = f'Bearer {self.api_key}'

        # Set the model ID based on the model name
        self.model_id = os.path.basename(self.model)

        # Set the URL based on the dataset type
        if self.api.startswith('local'):
            if self.dataset.startswith('speed_benchmark'):
                self.url = f'http://127.0.0.1:{self.port}/v1/completions'
            else:
                self.url = f'http://127.0.0.1:{self.port}/v1/chat/completions'

        # Set the apply_chat_template flag based on the URL
        if self.apply_chat_template is None:
            self.apply_chat_template = self.url.strip('/').endswith('chat/completions')

        # Set number and parallel to lists if they are integers
        if isinstance(self.number, int):
            self.number = [self.number]
        if isinstance(self.parallel, int):
            self.parallel = [self.parallel]
        assert len(self.number) == len(
            self.parallel
        ), f'The length of number and parallel should be the same, but got number: {self.number} and parallel: {self.parallel}'  # noqa: E501

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
                kv_dict = {}
                for kv in values:
                    parts = kv.split('=', 1)  # only split the first '='
                    if len(parts) != 2:
                        raise ValueError(f'Invalid key-value pair: {kv}')
                    key, value = parts
                    kv_dict[key.strip()] = value.strip()
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
    parser.add_argument('--api-key', type=str, required=False, default=None, help='The API key for authentication')
    parser.add_argument('--connect-timeout', type=int, default=600, help='The network connection timeout')
    parser.add_argument('--read-timeout', type=int, default=600, help='The network read timeout')
    parser.add_argument('--no-test-connection', action='store_false', default=False, help='Do not test the connection before starting the benchmark')  # noqa: E501

    # Performance and parallelism
    parser.add_argument('-n', '--number', type=int, default=1000, nargs='+', help='How many requests to be made')
    parser.add_argument('--parallel', type=int, default=1, nargs='+', help='Set number of concurrency requests, default 1')  # noqa: E501
    parser.add_argument('--rate', type=int, default=-1, help='Number of requests per second. default None')

    # Logging and debugging
    parser.add_argument('--log-every-n-query', type=int, default=10, help='Logging every n query')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug request send')
    parser.add_argument('--wandb-api-key', type=str, default=None, help='The wandb API key')
    parser.add_argument('--swanlab-api-key', type=str, default=None, help='The swanlab API key')
    parser.add_argument('--name', type=str, help='The wandb/swanlab db result name and result db name')

    # Prompt settings
    parser.add_argument('--max-prompt-length', type=int, default=sys.maxsize, help='Maximum input prompt length')
    parser.add_argument('--min-prompt-length', type=int, default=0, help='Minimum input prompt length')
    parser.add_argument('--prefix-length', type=int, default=0, help='The prefix length')
    parser.add_argument('--prompt', type=str, required=False, default=None, help='Specified the request prompt')
    parser.add_argument('--query-template', type=str, default=None, help='Specify the query template')
    parser.add_argument(
        '--apply-chat-template', type=argparse.BooleanOptionalAction, default=None, help='Apply chat template to the prompt')  # noqa: E501

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
    parser.add_argument('--seed', type=int, help='The random seed', default=0)
    parser.add_argument('--stop', nargs='*', help='The stop tokens', default=None)
    parser.add_argument('--stop-token-ids', nargs='*', help='Set the stop token IDs', default=None)
    parser.add_argument('--stream', action=argparse.BooleanOptionalAction, help='Stream output with SSE', default=True)
    parser.add_argument('--temperature', type=float, help='The sample temperature', default=0.0)
    parser.add_argument('--top-p', type=float, help='Sampling top p', default=None)
    parser.add_argument('--top-k', type=int, help='Sampling top k', default=None)
    parser.add_argument('--extra-args', type=json.loads, default='{}', help='Extra arguments, should in JSON format',)
    # yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark LLM service performance.')
    add_argument(parser)
    return parser.parse_args()
