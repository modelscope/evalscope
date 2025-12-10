import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, messages_pretty_str
from evalscope.api.metric import Score
from evalscope.api.model import Model
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = [
    'Chinese',
    'English',
    'German',
    'Italian',
    'Vietnamese',
    'Spanish',
    'Hindi',
    'Portuguese',
    'French',
    'Thai',
    'Russian',
]


@register_benchmark(
    BenchmarkMeta(
        name='multi_if',
        pretty_name='Multi-IF',
        description=
        'Multi-IF is a benchmark designed to evaluate the performance of LLM models\' capabilities in multi-turn instruction following within a multilingual environment.',  # noqa: E501
        tags=[Tags.INSTRUCTION_FOLLOWING, Tags.MULTI_LINGUAL, Tags.MULTI_TURN],
        dataset_id='facebook/Multi-IF',
        subset_list=SUBSET_LIST,
        metric_list=[
            'prompt_level_strict',
            'inst_level_strict',
            'prompt_level_loose',
            'inst_level_loose',
        ],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        extra_params={
            'max_turns': {
                'type': 'int',
                'description': 'Maximum number of interactive turns to evaluate (1-3).',
                'value': 3,
                'choices': [1, 2, 3]
            }
        }
    )
)
class MultiIFAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Ensure required packages are installed
        check_import(
            module_name=['nltk', 'langdetect'],
            package=['nltk', 'langdetect'],
            raise_error=True,
            feature_name=self.pretty_name
        )
        if 'Chinese' in self.subset_list:
            check_import(module_name='emoji', package='emoji', raise_error=True, feature_name='Chinese subset')
        if 'Thai' in self.subset_list:
            check_import(module_name='pythainlp', package='pythainlp', raise_error=True, feature_name='Thai subset')

        self.reformat_subset = True
        self.max_turns = self.extra_params.get('max_turns', 3)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=[ChatMessageUser(content='')],  # NOTE: we will build the multi turn conversation in the evaluator
            target='',
            subset_key=record['language'],
            metadata=record,
        )

    def run_inference(self, model: Model, sample: Sample, output_dir: str, **kwargs) -> TaskState:
        """
        Run multi-turn inference with the model and sample.
        """
        record = sample.metadata
        history = []
        step_record = {}
        for step in range(1, self.max_turns + 1):
            if not record.get(f'turn_{step}_prompt'):
                break
            current_prompt = json.loads(record[f'turn_{step}_prompt'])
            history.append(ChatMessageUser(content=current_prompt['content']))
            # Generate model output
            model_output = model.generate(input=history, tools=sample.tools)

            response = model_output.completion
            instruction_id_list = json.loads(record[f'turn_{step}_instruction_id_list'])
            kwargs_list = json.loads(record[f'turn_{step}_kwargs'])
            _kwargs = [json.loads(kwarg) for kwarg in kwargs_list]

            step_record[step] = {
                'prompt': messages_pretty_str(history),
                'response': response,
                'instruction_id_list': instruction_id_list,
                'kwargs': _kwargs
            }

            # Append model output to history for next turn
            history.append(model_output.message)

        sample.metadata['step_record'] = step_record
        return TaskState(
            model=model.name,
            sample=sample,
            messages=history,
            output=model_output,
            completed=True,
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: Dict, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        from .metrics import gen_acc_loose, gen_acc_strict, parse_result

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        step_record = task_state.metadata['step_record']
        results = {}
        try:
            for step, record in step_record.items():
                outputs_strict = gen_acc_strict(record)
                outputs_loose = gen_acc_loose(record)
                prompt_level_strict, inst_level_strict = parse_result([outputs_strict])
                prompt_level_loose, inst_level_loose = parse_result([outputs_loose])
                results.update({
                    f'turn_{step}_prompt_level_strict': prompt_level_strict,
                    f'turn_{step}_inst_level_strict': inst_level_strict,
                    f'turn_{step}_prompt_level_loose': prompt_level_loose,
                    f'turn_{step}_inst_level_loose': inst_level_loose,
                })
            score.value.update(results)

            # Set main score name
            if results:
                score.main_score_name = f'turn_{step}_prompt_level_strict'

        except Exception as e:
            logger.error(f'Error calculating ifeval metrics: {e}')
            score.value = {}

        return score
