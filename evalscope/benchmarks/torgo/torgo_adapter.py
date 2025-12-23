from typing import List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='torgo',
        pretty_name='TORGO',
        dataset_id='extraordinarylab/torgo',
        tags=[Tags.AUDIO, Tags.SPEECH_RECOGNITION],
        description=(
            'The TORGO database of dysarthric articulation consists of aligned acoustics and '
            'measured 3D articulatory features from speakers with either cerebral palsy (CP) '
            'or amyotrophic lateral sclerosis (ALS).'
        ),
        eval_split='test',
        subset_list=['mild', 'moderate', 'severe'],
        few_shot_num=0,
        metric_list=['cer', 'wer', 'sem_score'],
        prompt_template='Please recognize the speech and only output the recognized content:',
    )
)
class TorgoAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.add_overall_metric = False
        self.add_aggregation_name = False
        self.use_batch_scoring = True

        self.jiwer_cer = None
        self.jiwer_wer = None
        self.normalize_text = None
        self.sem_scorer = None

        if self.has_metric('cer') or self.has_metric('wer'):
            check_import('jiwer', 'jiwer', raise_error=True, feature_name='CER/WER Metric')
            try:
                if self.has_metric('cer'):
                    from jiwer import cer as jiwer_cer
                    self.jiwer_cer = jiwer_cer

                if self.has_metric('wer'):
                    from jiwer import wer as jiwer_wer
                    self.jiwer_wer = jiwer_wer

                from evalscope.metrics.text_normalizer.wer import normalize_text
                self.normalize_text = normalize_text
            except Exception as e:
                logger.warning(f'[TorgoAdapter] Failed to import jiwer components: {e}')

        if self.has_metric('sem_score'):
            check_import('jellyfish', 'jellyfish', raise_error=True, feature_name='SemScore Metric')
            try:
                from evalscope.metrics.metric import SemScore
                self.sem_scorer = SemScore()
            except Exception as e:
                logger.warning(f'[TorgoAdapter] Failed to initialize SemScore: {e}')

    def record_to_sample(self, record) -> Sample:
        content_list = [ContentText(text=self.prompt_template)]
        audio_base64 = bytes_to_base64(record['audio']['bytes'], format='wav', add_header=True, content_type='audio')
        content_list.append(ContentAudio(audio=audio_base64, format='wav'))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['transcript'],
            subset_key=record['intelligibility'],
            metadata={
                'transcript': record['transcript'],
                'intelligibility': record['intelligibility'],
                'duration': record['duration'],
            }
        )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        from evalscope.metrics.text_normalizer.wer import normalize_text

        language = 'en'

        normalized_prediction = normalize_text(original_prediction, language)
        normalized_reference = normalize_text(reference, language)
        score = Score(
            extracted_prediction=normalized_prediction,
            prediction=original_prediction,
        )

        cer_score = self.jiwer_cer(normalized_reference, normalized_prediction)
        wer_score = self.jiwer_wer(normalized_reference, normalized_prediction)
        score.value = {'cer': cer_score, 'wer': wer_score}
        return score

    def batch_match_score(
        self,
        original_predictions: List[str],
        filtered_predictions: List[str],
        references: List[str],
        task_states: List[TaskState],
    ) -> List[Score]:
        """Compute batched ASR metrics (CER, WER, SemScore)."""
        scores: List[Score] = []
        for i in range(len(original_predictions)):
            score = Score(
                extracted_prediction=filtered_predictions[i],
                prediction=original_predictions[i],
                value={},
            )
            scores.append(score)

        # ---- SemScore ----
        if self.has_metric('sem_score') and self.sem_scorer:
            try:
                sem_score = self.sem_scorer.apply(filtered_predictions, references)
                for i in range(len(scores)):
                    scores[i].value.update({'sem_score': sem_score[i]})
            except Exception as e:
                logger.warning(f'[TorgoAdapter] SemScore batch calculation failed: {e}')

        return scores
