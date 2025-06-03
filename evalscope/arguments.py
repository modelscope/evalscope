import argparse
import json

from evalscope.constants import EvalBackend, EvalStage, EvalType, JudgeStrategy, ModelTask, OutputType


class ParseStrArgsAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        assert isinstance(values, str), 'args should be a string.'

        # try json load first
        try:
            arg_dict = json.loads(values)
            setattr(namespace, self.dest, arg_dict)
            return
        except (json.JSONDecodeError, ValueError):
            pass

        # If JSON load fails, fall back to parsing as key=value pairs
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
    # yapf: disable
    # Model-related arguments
    parser.add_argument('--model', type=str, required=False, help='The model id on modelscope, or local model dir.')
    parser.add_argument('--model-id', type=str, required=False, help='The model id for model name in report.')
    parser.add_argument('--model-args', type=str, action=ParseStrArgsAction, help='The model args, should be a string.')
    parser.add_argument('--model-task', type=str, default=ModelTask.TEXT_GENERATION, choices=[ModelTask.TEXT_GENERATION, ModelTask.IMAGE_GENERATION], help='The model task for model id.')  # noqa: E501

    # Template-related arguments
    parser.add_argument('--template-type', type=str, required=False, help='Deprecated, will be removed in v1.0.0.')
    parser.add_argument('--chat-template', type=str, required=False, help='The custom jinja template for chat generation.')  # noqa: E501

    # Dataset-related arguments
    parser.add_argument('--datasets', type=str, nargs='+', required=False, help='Dataset id list, align to the module name in evalscope.benchmarks')  # noqa: E501
    parser.add_argument('--dataset-args', type=json.loads, default='{}', help='The dataset args, should be a json string.')  # noqa: E501
    parser.add_argument('--dataset-dir', help='The datasets dir.')
    parser.add_argument('--dataset-hub', help='The datasets hub.')

    # Generation configuration arguments
    parser.add_argument('--generation-config', type=str, action=ParseStrArgsAction, help='The generation config, should be a string.')  # noqa: E501

    # Evaluation-related arguments
    parser.add_argument('--eval-type', type=str, help='The type for evaluating.',
                        choices=[EvalType.CHECKPOINT, EvalType.CUSTOM, EvalType.SERVICE])
    parser.add_argument('--eval-backend', type=str, help='The evaluation backend to use.',
                        choices=[EvalBackend.NATIVE, EvalBackend.OPEN_COMPASS, EvalBackend.VLM_EVAL_KIT, EvalBackend.RAG_EVAL])  # noqa: E501
    parser.add_argument('--eval-config', type=str, required=False, help='The eval task config file path for evaluation backend.')  # noqa: E501
    parser.add_argument('--stage', type=str, default='all', help='The stage of evaluation pipeline.',
                        choices=[EvalStage.ALL, EvalStage.INFER, EvalStage.REVIEW])
    parser.add_argument('--limit', type=float, default=None, help='Max evaluation samples num for each subset.')
    parser.add_argument('--eval-batch-size', type=int, default=1, help='The batch size for evaluation.')

    # Cache and working directory arguments
    parser.add_argument('--mem-cache', action='store_true', default=False, help='Deprecated, will be removed in v1.0.0.')  # noqa: E501
    parser.add_argument('--use-cache', type=str, help='Path to reuse the cached results.')
    parser.add_argument('--work-dir', type=str, help='The root cache dir.')

    # Debug and runtime mode arguments
    parser.add_argument('--ignore-errors', action='store_true', default=False, help='Ignore errors during evaluation.')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode, will print information for debugging.')  # noqa: E501
    parser.add_argument('--dry-run', action='store_true', default=False, help='Dry run in single processing mode.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--api-key', type=str, default='EMPTY', help='The API key for the remote API model.')
    parser.add_argument('--api-url', type=str, default=None, help='The API url for the remote API model.')
    parser.add_argument('--timeout', type=float, default=None, help='The timeout for the remote API model.')
    parser.add_argument('--stream', action='store_true', default=False, help='Stream mode.')  # noqa: E501

    # LLMJudge arguments
    parser.add_argument('--judge-strategy', type=str, default=JudgeStrategy.AUTO, help='The judge strategy.')
    parser.add_argument('--judge-model-args', type=json.loads, default='{}', help='The judge model args, should be a json string.')  # noqa: E501
    parser.add_argument('--judge-worker-num', type=int, default=1, help='The number of workers for the judge model.')
    parser.add_argument('--analysis-report', action='store_true', default=False, help='Generate analysis report for the evaluation results using judge model.')  # noqa: E501
    # yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation on benchmarks for LLMs.')
    add_argument(parser)

    args = parser.parse_args()
    return args
