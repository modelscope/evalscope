import re
import json
from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import Choices, TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import prompt

SUBSET_LIST = [
    "analytical_chemistry",
    "general_chemistry",
    "materials_science",
    "physical_chemistry",
    "technical_chemistry",
    "chemical_preference",
    "inorganic_chemistry",
    "organic_chemistry",
    "toxicity_and_safety"
]

QA_TEMPLATE = """{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."""

@register_benchmark(
    BenchmarkMeta(
        name='chembench',
        pretty_name='ChemBench',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=
        "",  # noqa: E501
        dataset_id='jablonkagroup/ChemBench',
        metric_list=['exact_match'],
        subset_list=SUBSET_LIST,
        default_subset='all',
        few_shot_num=0,
        eval_split='train',
        prompt_template="""Answer the following multiple choice question. 
{question}

{choices}

Please reason step by step, and put your final answer within \\boxed{{}}. Your answer should be one of {letters}. """,
    )
)

class ChemBenchAdapter(DefaultDataAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    def record_to_sample(self, record) -> Sample:
        examples = record['examples'][0]
        is_mcq = False
        choices = None
        if examples['target_scores'] != None:
            answer = 0
            is_mcq = True
            target_scores = json.loads(examples['target_scores'])
            for k, v in target_scores.items():
                if v == 1:
                    break
                answer += 1
            answer = self.choices[answer]
            choices = list(target_scores.keys())
        else:
            answer = examples['target']

        return Sample(
            input=examples['input'],
            choices=choices if choices else None,
            target=answer,
            metadata={
                'is_mcq': is_mcq,
                'description': record['description'],
                'subject': record['subfield'],
                'keywords': record['keywords']
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Format the basic prompt template with the sample data.

        This method applies the prompt template to format the input text
        for models when no few-shot examples are used.

        Args:
            sample (Sample): The sample object containing the prompt data

        Returns:
            str: The formatted prompt ready for model input
        """
        if sample.metadata['is_mcq']:
            return prompt(
                question=sample.input,
                choices=Choices(sample.choices),
                template=self.prompt_template,
            )
        else:
            return QA_TEMPLATE.format(
                question=sample.input
            )
    
    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """
        Hook method for custom answer extraction from model predictions.

        This method can be overridden in subclasses to implement specific
        logic for extracting the final answer from complex model outputs.

        Args:
            prediction (str): The model prediction to extract from
            task_state (TaskState): The task state for additional context

        Returns:
            str: The extracted answer
        """
        if not prediction or not isinstance(prediction, str):
            return str(prediction)
        pattern = r"\\boxed\{([^}]*)\}"
        match = re.search(pattern, prediction)
        if match:
            return match.group(1)
        return prediction