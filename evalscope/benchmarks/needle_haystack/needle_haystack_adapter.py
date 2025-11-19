import os
from itertools import product
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, DictDataLoader, MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

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


@register_benchmark(
    BenchmarkMeta(
        name='needle_haystack',
        pretty_name='Needle-in-a-Haystack',
        tags=[Tags.RETRIEVAL, Tags.LONG_CONTEXT],
        description='Needle in a Haystack is a benchmark focused on information retrieval tasks. '
        'It requires the model to find specific information within a large corpus of text. '
        '[Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/needle_haystack.html)',  # noqa: E501
        dataset_id='AI-ModelScope/Needle-in-a-Haystack-Corpus',
        metric_list=['acc'],
        subset_list=['english', 'chinese'],
        eval_split='test',
        system_prompt='You are a helpful AI bot that answers questions for a user. Keep your response short and direct',
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            'retrieval_question': {
                'type': 'str',
                'description': 'Question used for retrieval evaluation.',
                'value': 'What is the best thing to do in San Francisco?'
            },
            'needles': {
                'type':
                'list[str]',
                'description':
                'List of factual needle strings inserted into the context.',
                'value':
                ['\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n']
            },
            'context_lengths_min': {
                'type': 'int',
                'description': 'Minimum context length (tokens) to generate synthetic samples.',
                'value': 1000
            },
            'context_lengths_max': {
                'type': 'int',
                'description': 'Maximum context length (tokens) to generate synthetic samples.',
                'value': 32000
            },
            'context_lengths_num_intervals': {
                'type': 'int',
                'description': 'Number of intervals between min and max context lengths.',
                'value': 10
            },
            'document_depth_percent_min': {
                'type': 'int',
                'description': 'Minimum insertion depth percentage for needles.',
                'value': 0
            },
            'document_depth_percent_max': {
                'type': 'int',
                'description': 'Maximum insertion depth percentage for needles.',
                'value': 100
            },
            'document_depth_percent_intervals': {
                'type': 'int',
                'description': 'Number of intervals between min and max depth percentages.',
                'value': 10
            },
            'tokenizer_path': {
                'type': 'str',
                'description': 'Tokenizer checkpoint path used for tokenization.',
                'value': 'Qwen/Qwen3-0.6B'
            },
            'show_score': {
                'type': 'bool',
                'description': 'Render numerical scores on heatmap output images.',
                'value': False
            }
        }
    )
)
class NeedleHaystackAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._use_llm_judge = True
        self.add_aggregation_name = False  # Don't add aggregation name for needle haystack adapter
        # set extra params
        self.retrieval_question = self.extra_params.get(
            'retrieval_question', 'What is the best thing to do in San Francisco?'
        )
        self.needles = self.extra_params.get(
            'needles',
            ['\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n']
        )
        self.context_lengths_min = self.extra_params.get('context_lengths_min', 1000)
        self.context_lengths_max = self.extra_params.get('context_lengths_max', 32000)
        self.context_lengths_num_intervals = self.extra_params.get('context_lengths_num_intervals', 10)
        self.document_depth_percent_min = self.extra_params.get('document_depth_percent_min', 0)
        self.document_depth_percent_max = self.extra_params.get('document_depth_percent_max', 100)
        self.document_depth_percent_intervals = self.extra_params.get('document_depth_percent_intervals', 10)
        self.tokenizer_path = self.extra_params.get('tokenizer_path', 'Qwen/Qwen3-0.6B')
        self.show_score = self.extra_params.get('show_score', False)

    def _init_length(self):
        """ Initialize context lengths and document depth percentages based on the provided parameters."""
        import numpy as np

        self.context_lengths = np.round(
            np.linspace(
                self.context_lengths_min,
                self.context_lengths_max,
                num=self.context_lengths_num_intervals,
                endpoint=True
            )
        ).astype(int)

        self.document_depth_percents = np.round(
            np.linspace(
                self.document_depth_percent_min,
                self.document_depth_percent_max,
                num=self.document_depth_percent_intervals,
                endpoint=True
            )
        ).astype(int)

    def _init_tokenizer(self):
        """ Initialize the tokenizer based on the provided tokenizer path."""
        from modelscope import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def load(self):
        """Load dataset from local disk or remote."""
        self._init_tokenizer()
        self._init_length()

        dataset_name_or_path = self.dataset_id
        if os.path.exists(dataset_name_or_path):
            logger.info(f'Loading dataset from {dataset_name_or_path}')
            dataset_path = dataset_name_or_path
        else:
            from modelscope import dataset_snapshot_download
            logger.info(f'Loading dataset from modelscope: > dataset_name: {dataset_name_or_path}')
            dataset_path = dataset_snapshot_download(
                dataset_name_or_path, allow_file_pattern=['PaulGraham_Essays.txt', 'Journey_to_the_West.txt']
            )

        # Load datasets for both subsets
        datasets = {}
        file_structure = {'english': ['PaulGraham_Essays.txt'], 'chinese': ['Journey_to_the_West.txt']}

        for subset_name, files in file_structure.items():
            if subset_name not in self.subset_list:
                continue
            file_path = os.path.join(dataset_path, files[0])
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Generate samples for all combinations of context length and depth
                records = []
                tokens_context = self._get_context_tokens(text)
                for context_length, depth_percent in tqdm(
                    product(self.context_lengths, self.document_depth_percents),
                    desc=f'Generating {subset_name} samples'
                ):
                    context = self._insert_needles(tokens_context, depth_percent, context_length)
                    record = {
                        'text': text,
                        'context_length': int(context_length),
                        'depth_percent': int(depth_percent),
                        'question': self.retrieval_question,
                        'answer': '\n'.join(self.needles),
                        'context': context,
                    }
                    records.append(record)

                dataset = DictDataLoader(
                    dict_list=records,
                    limit=self.limit,
                    repeats=self.repeats,
                    sample_fields=self.record_to_sample,
                    shuffle=self.shuffle,
                ).load()

                datasets[subset_name] = dataset

        test_dataset = DatasetDict(datasets)
        return test_dataset, None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        return Sample(
            input=record['question'],
            target=record['answer'],
            metadata={
                'context': record['context'],
                'context_length': record['context_length'],
                'depth_percent': record['depth_percent'],
            }
        )

    def format_prompt_template(self, sample):
        """Format the prompt template with context and question."""
        context = sample.metadata['context']
        question = sample.input
        return self.prompt_template.format(context=context, question=question)

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
                    '。'
                )  # Handle both English and Chinese periods

                # Then we iteration backwards until we find the first period
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]

                # Insert the needle into the context at the found position
                tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]

                # Log
                insertion_percentage = (insertion_point / len(tokens_context)) * 100
                self.insertion_percentages.append(insertion_percentage)
                logger.debug(
                    f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, "
                    f'total length now: {len(tokens_context)} tokens'
                )

                # Adjust depth for next needle
                depth_percent += depth_percent_interval

        new_context = self.tokenizer.decode(tokens_context)
        return new_context

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Calculate evaluation scores by comparing prediction with reference."""
        from evalscope.metrics import exact_match
        from .utils import normalize_answer

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Get metadata from task state
        context_length = task_state.metadata.get('context_length', 0)
        depth_percent = task_state.metadata.get('depth_percent', 0)

        norm_gold = normalize_answer(reference)
        norm_pred = normalize_answer(filtered_prediction)
        accuracy = exact_match(gold=norm_gold, pred=norm_pred)

        metric_name = f'Context#{context_length} Depth#{depth_percent}'
        score.value = {metric_name: accuracy}
        score.main_score_name = metric_name

        return score

    def llm_match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Use LLM as a judge to evaluate the predicted answer against the gold answer."""
        from .utils import GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE, parse_score

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Get metadata from task state
        context_length = task_state.metadata.get('context_length', 0)
        depth_percent = task_state.metadata.get('depth_percent', 0)
        question = task_state.input_text

        # Get grading response
        prompt = ORM_USER_TEMPLATE.format(question=question, gold=reference, pred=filtered_prediction)
        orm_response = self.llm_judge.judge(prompt, system_prompt=GENERAL_ORM_PROMPT)

        # Parse grading score with regex, [[score]]
        accuracy = parse_score(orm_response) if orm_response else 0.0

        metric_name = f'Context#{context_length} Depth#{depth_percent}'
        score.value = {metric_name: accuracy}
        score.explanation = f'LLM judge: {orm_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': getattr(self, 'judge_strategy', 'default'),
            'model': self.llm_judge.model_id if hasattr(self.llm_judge, 'model_id') else 'unknown'
        }
        score.main_score_name = metric_name

        return score

    def _on_generate_report_end(self, report: 'Report', output_dir: str, **kwargs):
        try:
            import os

            from .utils import draw_score_chat

            report_path = output_dir
            data_frame = report.to_dataframe()
            # split `Metric` to `Context` and `Depth`
            data_frame[['Context', 'Depth']] = data_frame['Metric'].str.split(' ', n=1, expand=True)
            data_frame['Depth'] = data_frame['Depth'].str.replace('Depth#', '').astype(float)
            data_frame['Context'] = data_frame['Context'].str.replace('Context#', '').astype(int)
            # split by `Subset` to multi sub data frame
            for subset in data_frame['Subset'].unique():
                sub_df = data_frame[data_frame['Subset'] == subset]
                # draw charts for each subset
                pivot_table = sub_df.pivot_table(values='Score', index=['Depth', 'Context'],
                                                 aggfunc='mean').reset_index()
                pivot_table = pivot_table.pivot(index='Depth', columns='Context', values='Score')
                draw_score_chat(
                    pivot_table,
                    outpath=os.path.join(report_path, f'needle_haystack_heatmap_{subset}.png'),
                    show_score=self.show_score
                )

        except Exception as e:
            logger.error(f'Error generating charts: {e}')
