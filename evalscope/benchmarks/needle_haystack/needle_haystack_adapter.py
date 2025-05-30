from itertools import product

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import AnswerKeys, EvalType
from evalscope.metrics import LLMJudge, exact_match

PROMPT_TEMPLATE = """Please read the following text and answer the question below.

<text>
{context}
</text>

<question>
{question}
</question>

Don't give information outside the document or repeat your findings."""


@Benchmark.register(
    name='needle_haystack',
    pretty_name='Needle in a Haystack',
    description='Needle in a Haystack is a benchmark focused on information retrieval tasks. \
    It requires the model to find specific information within a large corpus of text.',
    dataset_id='AI-ModelScope/Needle-in-a-Haystack-Corpus',
    metric_list=['AveragePrecision'],
    subset_list=['english', 'chinese'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    system_prompt='You are a helpful AI bot that answers questions for a user. Keep your response short and direct',
    prompt_template=PROMPT_TEMPLATE,
    extra_params={
        'retrieval_question': '',
        'needles': [],
        'context_lengths_min': 1000,
        'context_lengths_max': 16000,
        'context_lengths_num_intervals': 10,
        'document_depth_percent_min': 0,
        'document_depth_percent_max': 100,
        'document_depth_percent_intervals': 10,
    })
class NeedleHaystackAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.llm_as_a_judge = True
        # set extra params
        extra_params = kwargs.get('extra_params', {})
        self.retrieval_question = extra_params.get('retrieval_question',
                                                   'What is the best thing to do in San Francisco?')
        self.needles = extra_params.get(
            'needles',
            ['\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n'])
        self.context_lengths_min = extra_params.get('context_lengths_min', 1000)
        self.context_lengths_max = extra_params.get('context_lengths_max', 16000)
        self.context_lengths_num_intervals = extra_params.get('context_lengths_num_intervals', 10)
        self.document_depth_percent_min = extra_params.get('document_depth_percent_min', 0)
        self.document_depth_percent_max = extra_params.get('document_depth_percent_max', 100)
        self.document_depth_percent_intervals = extra_params.get('document_depth_percent_intervals', 10)

        self.__init_length()

    def __init_length(self):
        """ Initialize context lengths and document depth percentages based on the provided parameters."""
        import numpy as np

        self.context_lengths = np.round(
            np.linspace(
                self.context_lengths_min,
                self.context_lengths_max,
                num=self.context_lengths_num_intervals,
                endpoint=True)).astype(int)

        self.document_depth_percents = np.round(
            np.linspace(
                self.document_depth_percent_min,
                self.document_depth_percent_max,
                num=self.document_depth_percent_intervals,
                endpoint=True)).astype(int)

    def load(self, **kwargs):
        # default load with snapshot
        kwargs['file_structure'] = {'english': ['PaulGraham_Essays.txt'], 'chinese': ['Journey_to_the_West.txt']}
        data_dict = super().load_with_snapshot(**kwargs)
        return data_dict

    def gen_prompts(self, data_dict: dict) -> dict:
        """
        Generate dataset prompts from raw input, unify the prompt format for different datasets.

        Args:
            data_dict: {'english': {'test': [sample_d_1, sample_d_2, ...]},
                        'chinese': {'test': [sample_d_1, sample_d_2, ...]}}

        Returns:
            {'subset_name': [prompt_d_1, prompt_d_2, ...]}
            prompt_d_i (dict): refer to the output of gen_prompt method.

        e.g. train -- few-shot data, test -- target dataset to evaluate.
        """
        res_dict: dict = {}
        tasks = []
        for context_length, depth_percent in product(self.context_lengths, self.document_depth_percents):
            task = self.bound_evaluate_and_log(context_length, depth_percent)
            tasks.append(task)

        for sub_name, sub_data_dict in data_dict.items():
            res_dict[sub_name] = []
            for sample_d in sub_data_dict[self.eval_split]:
                prompt_d = self.gen_prompt(input_d=sample_d, subset_name=sub_name, few_shot_list=[])
                prompt_d[AnswerKeys.RAW_INPUT] = sample_d
                res_dict[sub_name].append(prompt_d)

        return res_dict

    def gen_prompt(self, input_d, subset_name, few_shot_list, **kwargs):
        """
        Generate the prompt for each sample in the dataset.

        Args:
            input_d: The input data for the sample
            subset_name: The name of the subset
            few_shot_list: List of few-shot examples

        Returns:
            A dictionary containing the prompt data
        """
        context = input_d['context'] if 'context' in input_d else input_d.get('text', '')
        question = input_d['question'] if 'question' in input_d else self.retrieval_question

        prompt = self.prompt_template.format(context=context, question=question)

        return self.gen_prompt_data(prompt, system_prompt=self.system_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        if 'Answer' in input_d:
            return input_d['Answer']

        # If there's no predefined answer, assume the answer is in the needle
        # This is for testing with inserted needles
        for needle in self.needles:
            if needle in input_d.get('context', ''):
                # Return a key fact from the needle
                return needle.strip()

        return ''

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        response = result.replace('*', '')

        if 'the answer is' in response.lower():
            ans = response.lower().rsplit('the answer is', 1)[-1].strip().strip('.').strip()
        else:
            ans = response.strip()

        return ans

    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and the predicted answer.
        """
        return exact_match(gold=gold, pred=pred)

    def llm_match(self, gold: str, pred: str, judge: LLMJudge, **kwargs) -> float:
        """
        Use LLM as a judge to evaluate the predicted answer against the gold answer.
        """
        from .utils import GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE

        raw_input = kwargs.get('raw_input', None)
        question = raw_input.get('question', self.retrieval_question)

        # get grading response
        prompt = ORM_USER_TEMPLATE.format(problem=question, answer_1=gold, answer_2=pred)
        orm_response = judge(prompt=prompt, system_prompt=GENERAL_ORM_PROMPT)

        # parse grading response
        if 'YES' in orm_response:
            return 1.0
        else:
            return 0.0
