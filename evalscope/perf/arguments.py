import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Arguments:
    model: str
    url: str = 'localhost'
    connect_timeout: int = 120
    read_timeout: int = 120
    number: Optional[int] = None
    parallel: int = 1
    rate: int = -1
    log_every_n_query: int = 10
    headers: Optional[Dict[str, Any]] = field(default_factory=dict)
    wandb_api_key: Optional[str] = None
    name: Optional[str] = None
    debug: bool = False
    tokenizer_path: Optional[str] = None
    api: str = 'openai'
    max_prompt_length: int = sys.maxsize
    min_prompt_length: int = 0
    prompt: Optional[str] = None
    query_template: Optional[str] = None
    dataset: str = 'line_by_line'
    dataset_path: Optional[str] = None
    frequency_penalty: Optional[float] = None
    logprobs: Optional[bool] = None
    max_tokens: Optional[int] = None
    n_choices: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = field(default_factory=list)
    stop_token_ids: Optional[List[str]] = field(default_factory=list)
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    @staticmethod
    def from_args(args):
        return Arguments(
            model=args.model,
            url=args.url,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            number=args.number,
            parallel=args.parallel,
            rate=args.rate,
            log_every_n_query=args.log_every_n_query,
            headers=args.headers,
            wandb_api_key=args.wandb_api_key,
            name=args.name,
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
            n_choices=args.n_choices,
            seed=args.seed,
            stop=args.stop,
            stop_token_ids=args.stop_token_ids,
            stream=args.stream,
            temperature=args.temperature,
            top_p=args.top_p)


def process_number(input):
    try:
        return int(input)
    except ValueError:
        try:
            return float(input)
        except ValueError:
            return input


class ParseKVAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for each in values:
            try:
                key, value = each.split('=')
                if value.lower() == 'bool_true':
                    value = True
                if value.lower() == 'bool_false':
                    value = False
                value = process_number(value)
                getattr(namespace, self.dest)[key] = value
            except ValueError as ex:
                message = '\nTraceback: {}'.format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))


def add_argument(parser: argparse.ArgumentParser):
    # yapf: disable
    parser.add_argument(
        '--model', type=str, required=True, help='The test model name.')
    parser.add_argument('--url', type=str, default='localhost')
    parser.add_argument('--connect-timeout', type=int, default=120, help='The network connection timeout')
    parser.add_argument('--read-timeout', type=int, default=120, help='The network read timeout')
    parser.add_argument('-n', '--number', type=int, default=None, help='How many requests to be made')
    parser.add_argument('--parallel', type=int, default=1, help='Set number of concurrency request, default 1')
    parser.add_argument('--rate', type=int, default=-1, help='Number of requests per second. default None')
    parser.add_argument('--log-every-n-query', type=int, default=10, help='Logging every n query')
    parser.add_argument('--headers', nargs='+', dest='headers', action=ParseKVAction, help='Extra http headers')
    parser.add_argument('--wandb-api-key', type=str, default=None, help='The wandb api key')
    parser.add_argument('--name', type=str, help='The wandb db result name and result db name')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug request send')
    parser.add_argument('--tokenizer-path', type=str, required=False, default=None, help='Specify the tokenizer weight path')  # noqa: E501
    parser.add_argument('--api', type=str, default='openai', help='Specify the service api')
    parser.add_argument('--max-prompt-length', type=int, default=sys.maxsize, help='Maximum input prompt length')
    parser.add_argument('--min-prompt-length', type=int, default=0, help='Minimum input prompt length')
    parser.add_argument('--prompt', type=str, required=False, default=None, help='Specified the request prompt')
    parser.add_argument('--query-template', type=str, default=None, help='Specify the query template')
    parser.add_argument('--dataset', type=str, default='line_by_line', help='Specify the dataset')
    parser.add_argument('--dataset-path', type=str, required=False, help='Path to the dataset file')
    parser.add_argument('--frequency-penalty', type=float, help='The frequency_penalty value', default=None)
    parser.add_argument('--logprobs', action='store_true', help='The logprobs', default=None)
    parser.add_argument('--max-tokens', type=int, help='The maximum number of tokens can be generated', default=None)
    parser.add_argument('--n-choices', type=int, help='How many completion choices to generate', default=None)
    parser.add_argument('--seed', type=int, help='The random seed', default=None)
    parser.add_argument('--stop', nargs='*', help='The stop tokens', default=None)
    parser.add_argument('--stop-token-ids', nargs='*', help='Set the stop token ids', default=None)
    parser.add_argument('--stream', action='store_true', help='Stream output with SSE', default=None)
    parser.add_argument('--temperature', type=float, help='The sample temperature', default=None)
    parser.add_argument('--top-p', type=float, help='Sampling top p', default=None)
    # yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark LLM service performance.')
    add_argument(parser)
    return parser.parse_args()
