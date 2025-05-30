from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import LLMJudge, exact_match

TEMPLATE_0SHOT = """Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)"."""


@Benchmark.register(
    name='frames',
    pretty_name='FRAMES',
    description=
    'FRAMES is a comprehensive evaluation dataset designed to test the capabilities of Retrieval-Augmented Generation (RAG) systems across factuality, retrieval accuracy, and reasoning.',  # noqa: E501
    dataset_id='iic/frames',
    model_adapter=OutputType.GENERATION,
    output_types=[OutputType.GENERATION],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    prompt_template=TEMPLATE_0SHOT,
)
class FramesAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, **kwargs):
        # default load with snapshot
        kwargs['file_structure'] = {'default': ['test.jsonl']}
        data_dict = super().load_with_snapshot(**kwargs)
        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from input data.
        """
        context = '\n'.join([f"{i['title']}\n{i['text']}" for i in input_d['wiki_items']])
        question = input_d['Prompt']
        prompt = self.prompt_template.format(context=context, question=question)
        return self.gen_prompt_data(prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        return input_d['Answer']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        response = result.replace('*', '')

        if 'the answer is' in response:
            ans = response.rsplit('the answer is', 1)[-1].strip().strip('.').strip()
        else:
            ans = ''

        return ans

    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and the predicted answer.
        """
        from .utils import normalize_answer
        gold = normalize_answer(gold)
        pred = normalize_answer(pred)
        return exact_match(gold=gold, pred=pred)

    def llm_match(self, gold: str, pred: str, judge: LLMJudge, **kwargs) -> float:
        from .utils import GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE

        raw_input = kwargs.get('raw_input', None)
        question = raw_input['Prompt']
        # get grading response
        prompt = ORM_USER_TEMPLATE.format(problem=question, answer_1=gold, answer_2=pred)
        orm_response = judge(prompt=prompt, system_prompt=GENERAL_ORM_PROMPT)
        # parse grading response
        if 'YES' in orm_response:
            return 1.0
        else:
            return 0.0
