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
        metric_list=[{
            'cer': {}
        }, {
            'wer': {}
        }, {
            'sem_score': {
                'model_id_or_path': 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
            }
        }],
        prompt_template='Please recognize the speech and only output the recognized content:',
    )
)
class TorgoAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.use_batch_scoring = True

        if self.has_metric('cer') or self.has_metric('wer'):
            check_import('jiwer', 'jiwer', raise_error=True, feature_name='CER/WER Metric')
        if self.has_metric('sem_score'):
            check_import('jellyfish', 'jellyfish', raise_error=True, feature_name='SemScore Metric')

    def record_to_sample(self, record) -> Sample:
        content_list = [ContentText(text=self.prompt_template)]
        audio_base64 = bytes_to_base64(record['audio']['bytes'], format='wav', add_header=True, content_type='audio')
        content_list.append(ContentAudio(audio=audio_base64, format='wav'))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['transcript'],
            subset_key=record['intelligibility'],
            metadata={
                'intelligibility': record['intelligibility'],
                'duration': record['duration'],
            }
        )

    def batch_match_score(
        self,
        original_predictions: List[str],
        filtered_predictions: List[str],
        references: List[str],
        task_states: List[TaskState],
    ) -> List[Score]:
        """Compute batched ASR metrics (CER, WER, SemScore)."""
        language = 'en'

        scores: List[Score] = []
        for i in range(len(original_predictions)):
            score = Score(
                extracted_prediction=filtered_predictions[i],
                prediction=original_predictions[i],
                value={},
            )
            scores.append(score)

        # ---- CER (per-sample within batch) ----
        if self.has_metric('cer'):
            try:
                from jiwer import cer as jiwer_cer

                from evalscope.metrics.text_normalizer.wer import normalize_text

                for i in range(len(scores)):
                    normalized_prediction = normalize_text(filtered_predictions[i], language)
                    normalized_reference = normalize_text(references[i], language)
                    cer_results = jiwer_cer(normalized_reference, normalized_prediction)
                    scores[i].value.update(cer_results)
            except Exception as e:
                logger.warning(f'[TorgoAdapter] CER batch calculation failed: {e}')

        # ---- WER (per-sample within batch) ----
        if self.has_metric('wer'):
            try:
                from jiwer import wer as jiwer_wer

                for i in range(len(scores)):
                    normalized_prediction = normalize_text(filtered_predictions[i], language)
                    normalized_reference = normalize_text(references[i], language)
                    wer_results = jiwer_wer(normalized_reference, normalized_prediction)
                    scores[i].value.update(wer_results)
            except Exception as e:
                logger.warning(f'[TorgoAdapter] WER batch calculation failed: {e}')

        # ---- SemScore ----
        if self.has_metric('sem_score'):
            try:
                from evalscope.metrics.metric import SemScore

                score_args = self.get_metric_args('sem_score')
                sem_scorer = SemScore(**score_args)
                sem_score = sem_scorer.apply(filtered_predictions, references)
                for i in range(len(scores)):
                    scores[i].value.update({'sem_score': sem_score[i]})
            except Exception as e:
                logger.warning(f'[TorgoAdapter] SemScore batch calculation failed: {e}')
