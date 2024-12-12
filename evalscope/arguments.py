import argparse
import json

from evalscope.config import TaskConfig


class ParseStrArgsAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        assert isinstance(values, str), 'args should be a string.'

        arg_dict = {}
        for arg in values.strip().split(','):
            key, value = map(str.strip, arg.split('=', 1))  # Use maxsplit=1 to handle multiple '='
            try:
                # Safely evaluate the value using eval
                arg_dict[key] = eval(value)
            except Exception:
                # If eval fails, check if it's a boolean value
                value_lower = value.lower()
                if value_lower == 'true':
                    arg_dict[key] = True
                elif value_lower == 'false':
                    arg_dict[key] = False
                else:
                    # If not a boolean, keep the original string
                    arg_dict[key] = value

        setattr(namespace, self.dest, arg_dict)


def add_argument(parser: argparse.ArgumentParser):
    # Model-related arguments
    parser.add_argument('--model', type=str, required=False, help='The model id on modelscope, or local model dir.')
    parser.add_argument('--model-args', type=str, action=ParseStrArgsAction, help='The model args, should be a string.')

    # Template-related arguments
    parser.add_argument('--template-type', type=str, required=False, help='Deprecated, will be removed in v1.0.0.')
    parser.add_argument(
        '--chat-template', type=str, required=False,
        help='The custom jinja template for chat generation.')  # noqa: E501

    # Dataset-related arguments
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=False,
        help='Dataset id list, align to the module name in evalscope.benchmarks')  # noqa: E501
    parser.add_argument(
        '--dataset-args', type=json.loads, default='{}',
        help='The dataset args, should be a json string.')  # noqa: E501
    parser.add_argument('--dataset-dir', help='The datasets dir.')
    parser.add_argument('--dataset-hub', help='The datasets hub.')

    # Generation configuration arguments
    parser.add_argument(
        '--generation-config', type=str, action=ParseStrArgsAction,
        help='The generation config, should be a string.')  # noqa: E501

    # Evaluation-related arguments
    parser.add_argument('--eval-type', type=str, help='The type for evaluating.')
    parser.add_argument('--eval-backend', type=str, help='The evaluation backend to use.')
    parser.add_argument(
        '--eval-config', type=str, required=False,
        help='The eval task config file path for evaluation backend.')  # noqa: E501
    parser.add_argument('--stage', type=str, default='all', help='The stage of evaluation pipeline.')
    parser.add_argument('--limit', type=int, default=None, help='Max evaluation samples num for each subset.')

    # Cache and working directory arguments
    parser.add_argument(
        '--mem-cache', action='store_true', default=False, help='Deprecated, will be removed in v1.0.0.')  # noqa: E501
    parser.add_argument('--use-cache', type=str, help='Path to reuse the cached results.')
    parser.add_argument('--work-dir', type=str, help='The root cache dir.')

    # Debug and runtime mode arguments
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Debug mode, will print information for debugging.')  # noqa: E501
    parser.add_argument('--dry-run', action='store_true', default=False, help='Dry run in single processing mode.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    # yapf: enable


def parse_args():
    # yapf: disable
    parser = argparse.ArgumentParser(description='Run evaluation on benchmarks for LLMs.')
    add_argument(parser)

    args = parser.parse_args()
    return args


def convert_args(args):
    # Convert Namespace to a dictionary and filter out None values
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    del args_dict['func']  # Note: compact CLI arguments

    return TaskConfig(**args_dict)