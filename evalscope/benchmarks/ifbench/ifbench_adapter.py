from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='ifbench',
        pretty_name='IFBench',
        description=
        'IFBench is a new benchmark designed to evaluate how reliably AI models follow novel, challenging, and diverse verifiable instructions, with a strong focus on out-of-domain generalization. It comprises 58 manually curated verifiable constraints across categories such as counting, formatting, and word usage, aiming to address overfitting and data contamination issues present in existing benchmarks. Developed by AllenAI, IFBench serves as a rigorous test for precise instruction-following capabilities.',  # noqa: E501
        tags=[Tags.INSTRUCTION_FOLLOWING],
        dataset_id='allenai/IFBench_test',
        subset_list=['default'],
        metric_list=[
            'prompt_level_strict',
            'inst_level_strict',
            'prompt_level_loose',
            'inst_level_loose',
        ],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
    )
)
class IFBenchAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import(
            module_name=['emoji', 'syllapy', 'spacy'],
            package=['emoji', 'syllapy', 'spacy'],
            raise_error=True,
            feature_name=self.pretty_name
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompt = record.get('prompt', '')
        message_list = [ChatMessageUser(content=prompt)]

        return Sample(input=message_list, target='', metadata=record)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: Dict, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        from evalscope.benchmarks.ifbench.evaluation_lib import process_results

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        doc = task_state.metadata
        try:
            # Process results using the existing ifeval utility
            results = process_results(doc, [filtered_prediction])
            score.value.update(results)

            # Set main score name
            score.main_score_name = 'prompt_level_strict'

        except Exception as e:
            logger.error(f'Error calculating ifbench metrics: {e}')
            score.value = {}

        return score
