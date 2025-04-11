from tqdm import tqdm

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.utils.logger import get_logger

logger = get_logger()


@Benchmark.register(
    name='live_code_bench',
    pretty_name='Live Code Bench',
    dataset_id='AI-ModelScope/code_generation_lite',
    subset_list=['release_latest'],
    metric_list=['Pass@1'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    extra_params={
        'start_date': None,
        'end_date': None,
        'timeout': 6,
        'debug': False
    },
    system_prompt=
    'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.',  # noqa: E501
    prompt_template=
    '### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)\n\n',  # noqa: E501
)
class LiveCodeBenchAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        extra_params = kwargs.get('extra_params', {})

        self.timeout = extra_params.get('timeout', 6)
        self.debug = extra_params.get('debug', False)
        self.start_date = extra_params.get('start_date')
        self.end_date = extra_params.get('end_date')

    def load(self, **kwargs) -> dict:
        from .load_utils import filter_date, transform

        # Note: need trust_remote_code=True to load the python script
        dataset_dict = super().load(trust_remote_code=True, **kwargs)
        new_dataset_dict = {}
        for subset_key, dataset in dataset_dict.items():
            datasets = dataset[self.eval_split]
            filtered_datasets = filter_date(datasets, start_date=self.start_date, end_date=self.end_date)

            transformed_datasets = [transform(item) for item in tqdm(filtered_datasets, desc='Transforming data')]
            new_dataset_dict[subset_key] = {self.eval_split: transformed_datasets}
        return new_dataset_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate the prompt for the model input.
        """
        format_prompt = input_d['format_prompt']
        question_content = input_d['question_content']
        full_prompt = self.prompt_template.format(question_content=question_content, format_prompt=format_prompt)

        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        return input_d

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.
        """
        return result

    def match(self, gold: dict, pred: str) -> float:
        from .evaluate_utils import codegen_metrics
        from .extract_utils import extract_code_generation

        ext_pred = extract_code_generation(pred)

        references = [{'input_output': gold['evaluation_sample']}]
        predictions = [[ext_pred]]
        metrics, eval_results, final_metadata = codegen_metrics(
            references,
            predictions,
            k_list=[1],
            num_process_evaluate=1,
            timeout=self.timeout,
            debug=self.debug,
        )
        return metrics['pass@1'] / 100  # convert to point scale
