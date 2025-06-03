from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType
from evalscope.metrics import LLMJudge

TEMPLATE_0SHOT = """Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)"."""


@Benchmark.register(
    name='docmath',
    pretty_name='DocMath',
    description=
    'DocMath-Eval is a comprehensive benchmark focused on numerical reasoning within specialized domains. It requires the model to comprehend long and specialized documents and perform numerical reasoning to answer the given question.',  # noqa: E501
    dataset_id='yale-nlp/DocMath-Eval',
    metric_list=['AverageAccuracy'],
    subset_list=['complong_testmini', 'compshort_testmini', 'simplong_testmini', 'simpshort_testmini'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    prompt_template=TEMPLATE_0SHOT,
)
class DocMathAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, **kwargs):
        # default load mini test
        kwargs['split_as_subset'] = True
        data_dict = super().load(**kwargs)
        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from input data.
        """
        context = context = '\n'.join(input_d['paragraphs'])
        question = input_d['question']
        prompt = self.prompt_template.format(context=context, question=question)
        return self.gen_prompt_data(prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        return input_d['ground_truth']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        from .utils import extract_answer

        extracted_answer = extract_answer(result)
        return extracted_answer

    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and the predicted answer.
        """
        from .utils import get_acc

        return get_acc(prediction=pred, gt=gold)

    def llm_match(self, gold: str, pred: str, judge: LLMJudge, **kwargs) -> float:
        from .utils import GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE

        raw_input = kwargs.get('raw_input', None)
        question = raw_input['question']
        # get grading response
        prompt = ORM_USER_TEMPLATE.format(problem=question, answer_1=gold, answer_2=pred)
        orm_response = judge(prompt=prompt, system_prompt=GENERAL_ORM_PROMPT)
        # parse grading response
        if 'YES' in orm_response:
            return 1.0
        else:
            return 0.0
