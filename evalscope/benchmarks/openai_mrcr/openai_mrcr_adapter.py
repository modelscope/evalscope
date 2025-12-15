from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.metric.scorer import AggScore, SampleScore
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from .utils import OPENAI_MRCR_BINS, bin_index_for, get_chatml_tok_cnt, get_token_count, grade, str_to_chat_messages

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='openai_mrcr',
        pretty_name='OpenAI MRCR',
        tags=[Tags.LONG_CONTEXT, Tags.RETRIEVAL],
        description='Memory-Recall with Contextual Retrieval (MRCR). '
        'Evaluates retrieval and recall in long contexts by placing 2, 4 or 8 needles in the prompt. '
        'Measures whether the model can correctly extract and use them. ',
        dataset_id='openai-mirror/mrcr',
        metric_list=['mrcr_score'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template='',  # Not used, we use chat messages directly
        extra_params={
            'max_context_size': {
                'type': 'int | null',
                'description': 'Maximum context tokens; samples exceeding are skipped. Defaults to None (no limit).',
                'value': None
            },
            'min_context_size': {
                'type': 'int | null',
                'description': 'Minimum context tokens; samples below are skipped. Defaults to None (no limit).',
                'value': None
            },
            'needle_count': {
                'type': 'list[int] | null',
                'description':
                'Needle count filter (allowed: 2,4,8). Must be a list, e.g., [2], [4], or [2, 4, 8].  None keeps all.',
                'value': None
            },
            'tik_enc': {
                'type': 'str',
                'description': 'tiktoken encoding name used for token counting.',
                'value': 'o200k_base'
            },
            'prefix_filter': {
                'type': 'str | null',
                'description': 'Regex pattern to filter answers. Defaults to None (no filtering).',
                'value': None
            }
        }
    )
)
class OpenAIMRCRAdapter(DefaultDataAdapter):
    """Adapter for OpenAI MRCR benchmark.

    This benchmark evaluates long-context retrieval and recall by placing
    needles (key information) in long prompts and testing if the model
    can extract and use them correctly.

    The adapter automatically generates subset scores for each token count bin:
    - Overall: Average across all samples
    - 4096-8192: Samples with 4K-8K total tokens
    - 8192-16384: Samples with 8K-16K total tokens
    - 16384-32768: Samples with 16K-32K total tokens
    - 32768-65536: Samples with 32K-64K total tokens
    - 65536-131072: Samples with 64K-128K total tokens
    - 131072-262144: Samples with 128K-256K total tokens
    - 262144-524288: Samples with 256K-512K total tokens
    - 524288-1048576: Samples with 512K-1M total tokens
    """

    def __init__(self, **kwargs):
        """Initialize the MRCR adapter.

        Extra params:
            max_context_size (int, optional): Maximum context size in tokens.
                Samples exceeding this will be filtered out. Defaults to None (no limit).
            needle_count (list[int], optional): Filter by specific needle count(s) (2, 4, and/or 8).
                Must be a list, e.g., [2], [4], or [2, 4, 8]. Defaults to None (include all needle counts).
            min_context_size (int, optional): Keep only samples whose
                total token count is strictly greater than this value.
        """
        super().__init__(**kwargs)
        self.enc_name = self.extra_params.get('tik_enc', 'o200k_base')
        self.max_context_size = self.extra_params.get('max_context_size')
        self.needle_count = self.extra_params.get('needle_count')
        self.min_context_size = self.extra_params.get('min_context_size')
        self.prefix_filter = self.extra_params.get('prefix_filter', '\r\n ')

    def load(self):
        import tiktoken

        self.tik_enc = tiktoken.get_encoding(self.enc_name)
        if self.needle_count is not None:
            if not isinstance(self.needle_count, list):
                logger.warning('needle_count must be list; ignoring')
                self.needle_count = None
            else:
                bad = [n for n in self.needle_count if n not in (2, 4, 8)]
                if bad:
                    logger.warning(f'Invalid needle_count values {bad}; ignoring')
                    self.needle_count = None

        if self.max_context_size is not None:
            if not isinstance(self.max_context_size, int) or self.max_context_size < 0:
                logger.warning('max_context_size must be a non-negative integer; ignoring')
                self.max_context_size = None

        if self.min_context_size is not None:
            if not isinstance(self.min_context_size, int) or self.min_context_size < 0:
                logger.warning('min_context_size must be a non-negative integer; ignoring')
                self.min_context_size = None

        return super().load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a raw MRCR record to a Sample object.

        Expected fields in the source record:
        - prompt (str): JSON string containing chat messages
        - answer (str): expected output
        - random_string_to_prepend (str): prefix for exact matching
        - n_needles (int): number of needles in the context
        - desired_msg_index (int): index of the desired message
        - total_messages (int): total number of messages
        - n_chars (int): number of characters in context

        Args:
            record: Raw data record from the dataset

        Returns:
            Sample object or empty list if filtered out
        """
        # Filter by needle count EARLY (before expensive parsing/tokenizing)
        if self.needle_count is not None and record.get('n_needles') not in self.needle_count:
            return []
        input_tok_cnt = get_chatml_tok_cnt(record.get('prompt'), self.tik_enc)
        if self.max_context_size is not None and input_tok_cnt > self.max_context_size:
            return []
        if self.min_context_size is not None and input_tok_cnt <= self.min_context_size:
            return []
        output_tok_cnt = get_token_count(record.get('answer'), self.tik_enc)
        total_tok_cnt = input_tok_cnt + output_tok_cnt

        bin_index = bin_index_for(total_tok_cnt)

        metadata = {
            'random_string_to_prepend': record.get('random_string_to_prepend'),
            'n_needles': record.get('n_needles'),
            'desired_msg_index': record.get('desired_msg_index'),
            'total_messages': record.get('total_messages'),
            'n_chars': record.get('n_chars'),
            'raw_input_tok_cnt': input_tok_cnt,
            'bin_index': bin_index,
        }
        return Sample(input=str_to_chat_messages(record['prompt']), target=record['answer'], metadata=metadata)

    def filter_prediction(self, prediction: str, task_state: TaskState) -> str:
        """Strip stray newlines that some models emit before the MRCR prefix."""
        filtered = super().filter_prediction(prediction, task_state)
        return filtered.lstrip(self.prefix_filter)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Calculate MRCR-specific evaluation scores.

        This method computes the sequence ratio score using MRCR's exact
        grading logic with prefix handling, and assigns the sample to a
        token count bin for bucketed metrics.

        Args:
            original_prediction: The original, unfiltered model prediction
            filtered_prediction: The filtered and processed prediction
            reference: The ground truth reference answer
            task_state: The complete task state for context

        Returns:
            Score object containing the sequence ratio and bin metadata
        """
        prefix = task_state.metadata.get('random_string_to_prepend') if task_state.metadata else None

        # Calculate sequence ratio with MRCR prefix handling
        ratio = grade(prediction=filtered_prediction, reference=reference, random_string_to_prepend=prefix)

        bin_index = task_state.metadata.get('bin_index')
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        score.value['mrcr_score'] = ratio
        score.metadata['bin_index'] = bin_index
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Aggregate MRCR scores by token count bins.

        This method computes:
        1. Overall average mrcr_score across all samples
        2. Per-bin average mrcr_score for each token count range

        Each bin appears as a separate metric in the report (e.g., mrcr_score@4096-8192).
        Bins with no samples are automatically excluded from the results.

        Args:
            sample_scores: List of individual sample scores

        Returns:
            List of AggScore objects containing overall and per-bin metrics.
            Returns empty list if no valid scores are found.
        """
        if not sample_scores:
            return []
        bin_scores: Dict[int, List[float]] = {i: [] for i in range(len(OPENAI_MRCR_BINS))}
        for s in sample_scores:
            sc = s.score
            if not sc or not sc.metadata or not sc.value or 'mrcr_score' not in sc.value:
                continue
            idx = sc.metadata.get('bin_index')
            if idx is None or not (0 <= idx < len(OPENAI_MRCR_BINS)):
                continue
            bin_scores[idx].append(sc.value['mrcr_score'])
        overall = [v for vals in bin_scores.values() for v in vals]
        if not overall:
            return []
        agg: List[AggScore] = [
            AggScore(
                metric_name='mrcr_score',
                aggregation_name='overall',
                score=sum(overall) / len(overall),
                num=len(overall)
            )
        ]
        for i, vals in bin_scores.items():
            if not vals:
                continue
            l, r = OPENAI_MRCR_BINS[i]
            agg.append(
                AggScore(
                    metric_name='mrcr_score', aggregation_name=f'{l}-{r}', score=sum(vals) / len(vals), num=len(vals)
                )
            )
        return agg
