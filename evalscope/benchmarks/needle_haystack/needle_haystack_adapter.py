from itertools import product
from tqdm import tqdm
from typing import TYPE_CHECKING, List, Union

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import AnswerKeys, EvalType
from evalscope.metrics import LLMJudge, exact_match
from evalscope.metrics.metrics import mean
from evalscope.utils import get_logger

if TYPE_CHECKING:
    from evalscope.report import Report

logger = get_logger()

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
    metric_list=['AverageAccuracy'],
    subset_list=['english', 'chinese'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    system_prompt='You are a helpful AI bot that answers questions for a user. Keep your response short and direct',
    prompt_template=PROMPT_TEMPLATE,
    extra_params={
        'retrieval_question': 'What is the best thing to do in San Francisco?',
        'needles':
        ['\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n'],
        'context_lengths_min': 1000,
        'context_lengths_max': 32000,
        'context_lengths_num_intervals': 10,
        'document_depth_percent_min': 0,
        'document_depth_percent_max': 100,
        'document_depth_percent_intervals': 10,
        'tokenizer_path': 'Qwen/Qwen3-0.6B',
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
        self.context_lengths_max = extra_params.get('context_lengths_max', 32000)
        self.context_lengths_num_intervals = extra_params.get('context_lengths_num_intervals', 10)
        self.document_depth_percent_min = extra_params.get('document_depth_percent_min', 0)
        self.document_depth_percent_max = extra_params.get('document_depth_percent_max', 100)
        self.document_depth_percent_intervals = extra_params.get('document_depth_percent_intervals', 10)
        self.tokenizer_path = extra_params.get('tokenizer_path', 'Qwen/Qwen3-0.6B')

        self.__init_tokenizer()
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

    def __init_tokenizer(self):
        """ Initialize the tokenizer based on the provided tokenizer path."""
        from modelscope import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

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

        for sub_name, sub_data_dict in data_dict.items():
            res_dict[sub_name] = []
            for sample_d in sub_data_dict[self.eval_split]:
                # Generate prompts for each sample in the dataset
                tokens_context = self._get_context_tokens(sample_d['text'])
                for context_length, depth_percent in tqdm(
                        product(self.context_lengths, self.document_depth_percents),
                        desc=f'Generating {sub_name} prompts'):
                    # Insert needles into the context at the specified depth percentage
                    context = self._insert_needles(tokens_context, depth_percent, context_length)
                    # Build the input dictionary for the prompt
                    input_d = {
                        'context_length': int(context_length),
                        'depth_percent': int(depth_percent),
                        'question': self.retrieval_question,
                        'answer': '\n'.join(self.needles),
                        'context': context,
                    }
                    prompt_d = self.gen_prompt(input_d=input_d)
                    prompt_d[AnswerKeys.RAW_INPUT] = input_d
                    res_dict[sub_name].append(prompt_d)

        return res_dict

    def _get_context_tokens(self, input_context: str) -> list:
        """
        Encodes the context string into tokens using the tokenizer, ensuring the tokenized context
        is at least as long as the maximum context length required.

        Args:
            input_context (str): The context string to be tokenized.

        Returns:
            List[int]: A list of token IDs representing the context.
        """
        max_context_length = max(self.context_lengths)
        context = input_context
        tokens_context = self.tokenizer.encode(context, add_special_tokens=False)
        # Repeat the context until reaching the required length
        while len(tokens_context) < max_context_length:
            context += '\n' + input_context
            tokens_context = self.tokenizer.encode(context, add_special_tokens=False)
        return tokens_context

    def _insert_needles(self, tokens_context, depth_percent, context_length):
        """
        Inserts multiple needles (specific facts or pieces of information) into the original context string at
        designated depth percentages, effectively distributing these needles throughout the context. This method
        is designed to test a model's ability to retrieve specific information (needles) from a larger body of text
        (haystack) based on the placement depth of these needles.

        The method first encodes the context and each needle into tokens to calculate their lengths in tokens.
        It then adjusts the context length to accommodate the final buffer length. This is crucial for ensuring
        that the total token count (context plus needles) does not exceed the maximum allowable context length,
        which might otherwise lead to information being truncated.

        This approach calculates the initial insertion point for the first needle as before but then calculates even
        spacing for the remaining needles based on the remaining context length. It ensures that needles are
        distributed as evenly as possible throughout the context after the first insertion.

        Args:
            tokens_context (List[int]): The original context tokens.
            depth_percent (float): The depth percent at which to insert the needles.
            context_length (int): The total length of the context in tokens, adjusted for final buffer.

        Returns:
            str: The new context with needles inserted.
        """

        context_length -= 150

        # Calculate the total length of all needles in tokens
        total_needles_length = sum(len(self.tokenizer.encode(needle)) for needle in self.needles)

        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]

        # To evenly distribute the needles, we calculate the intervals they need to be inserted.
        depth_percent_interval = (100 - depth_percent) / len(self.needles)

        # Reset the insertion percentages list for the current context
        self.insertion_percentages = []

        # Insert needles at calculated points
        for needle in self.needles:

            tokens_needle = self.tokenizer.encode(needle)

            if depth_percent == 100:
                # If your depth percent is 100 (which means your needle is the last thing in the doc),
                # throw it at the end
                tokens_context = tokens_context + tokens_needle
            else:
                # Go get the position (in terms of tokens) to insert your needle
                insertion_point = int(len(tokens_context) * (depth_percent / 100))

                # tokens_new_context represents the tokens before the needle
                tokens_new_context = tokens_context[:insertion_point]

                # We want to make sure that we place our needle at a sentence break
                # so we first see what token a '.' is
                period_tokens = self.tokenizer.encode('.') + self.tokenizer.encode(
                    '。')  # Handle both English and Chinese periods

                # Then we iteration backwards until we find the first period
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]

                # Insert the needle into the context at the found position
                tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]

                # Log
                insertion_percentage = (insertion_point / len(tokens_context)) * 100
                self.insertion_percentages.append(insertion_percentage)
                logger.debug(f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, "
                             f'total length now: {len(tokens_context)} tokens')

                # Adjust depth for next needle
                depth_percent += depth_percent_interval

        new_context = self.tokenizer.decode(tokens_context)
        return new_context

    def gen_prompt(self, input_d: dict, **kwargs) -> dict:
        """
        Generate the prompt for each sample in the dataset.
        Args:
            input_d: A dictionary containing the input data for the prompt.
                It should contain 'context' and optionally 'question'.
        Returns:
            A dictionary containing the prompt data
        """
        context = input_d.get('context')
        question = input_d.get('question')

        prompt = self.prompt_template.format(context=context, question=question)

        return self.gen_prompt_data(prompt, system_prompt=self.system_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        return input_d.get('answer', '').strip()

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        return result

    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and the predicted answer.
        """
        from .utils import normalize_answer
        norm_gold = normalize_answer(gold)
        norm_pred = normalize_answer(pred)
        # Use exact match for Needle in a Haystack
        return exact_match(gold=norm_gold, pred=norm_pred)

    def llm_match(self, gold: str, pred: str, judge: LLMJudge, **kwargs) -> dict:
        """
        Use LLM as a judge to evaluate the predicted answer against the gold answer.
        """
        from .utils import GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE, parse_score

        raw_input = kwargs.get('raw_input', None)
        question = raw_input.get('question')
        context_length = raw_input.get('context_length')
        depth_percent = raw_input.get('depth_percent')

        # get grading response
        prompt = ORM_USER_TEMPLATE.format(question=question, gold=gold, pred=pred)
        orm_response = judge(prompt=prompt, system_prompt=GENERAL_ORM_PROMPT)

        # parse grading score with regex, [[score]]
        score = parse_score(orm_response) if orm_response else 0.0
        return {f'Context#{context_length} Depth#{depth_percent}': score}

    def compute_metric(self, review_res_list: Union[List[dict], List[List[dict]]], **kwargs) -> List[dict]:
        """
        compute weighted mean of the bleu score of all samples

        Args:
            review_res_list: [score1, score2, ...]

        Returns:
            avg_res: List[dict]

        """
        items = super().compute_dict_metric(review_res_list, **kwargs)
        return [{'metric_name': k, 'score': mean(v), 'num': len(v)} for k, v in items.items()]

    def post_process_report(self, report: 'Report', **kwargs):
        try:
            import os

            from .utils import draw_score_chat

            report_path = kwargs.get('report_path')
            data_frame = report.to_dataframe()
            # split `Metric` to `Context` and `Depth`
            data_frame[['Context', 'Depth']] = data_frame['Metric'].str.split(' ', n=1, expand=True)
            data_frame['Depth'] = data_frame['Depth'].str.replace('Depth#', '').astype(float)
            data_frame['Context'] = data_frame['Context'].str.replace('Context#', '').astype(int)
            # split by `Subset` to multi sub data frame
            for subset in data_frame['Subset'].unique():
                sub_df = data_frame[data_frame['Subset'] == subset]
                # draw charts for each subset
                pivot_table = sub_df.pivot_table(
                    values='Score', index=['Depth', 'Context'], aggfunc='mean').reset_index()
                pivot_table = pivot_table.pivot(index='Depth', columns='Context', values='Score')
                draw_score_chat(pivot_table, outpath=os.path.join(report_path, f'needle_haystack_heatmap_{subset}.png'))

        except Exception as e:
            logger.error(f'Error generating charts: {e}')
