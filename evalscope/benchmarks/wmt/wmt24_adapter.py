from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """
Translate the following {source_language} sentence into {target_language}:

{source_language}: {source_text}
{target_language}:
""".strip()

LANGUAGE_PAIRS = [
    'en-ar_eg',
    'en-ar_sa',
    'en-bg_bg',
    'en-bn_in',
    'en-ca_es',
    'en-cs_cz',
    'en-da_dk',
    'en-de_de',
    'en-el_gr',
    'en-es_mx',
    'en-et_ee',
    'en-fa_ir',
    'en-fi_fi',
    'en-fil_ph',
    'en-fr_ca',
    'en-fr_fr',
    'en-gu_in',
    'en-he_il',
    'en-hi_in',
    'en-hr_hr',
    'en-hu_hu',
    'en-id_id',
    'en-is_is',
    'en-it_it',
    'en-ja_jp',
    'en-kn_in',
    'en-ko_kr',
    'en-lt_lt',
    'en-lv_lv',
    'en-ml_in',
    'en-mr_in',
    'en-nl_nl',
    'en-no_no',
    'en-pa_in',
    'en-pl_pl',
    'en-pt_br',
    'en-pt_pt',
    'en-ro_ro',
    'en-ru_ru',
    'en-sk_sk',
    'en-sl_si',
    'en-sr_rs',
    'en-sv_se',
    'en-sw_ke',
    'en-sw_tz',
    'en-ta_in',
    'en-te_in',
    'en-th_th',
    'en-tr_tr',
    'en-uk_ua',
    'en-ur_pk',
    'en-vi_vn',
    'en-zh_cn',
    'en-zh_tw',
    'en-zu_za',
]

LANGUAGE_BY_CODE = {
    'ar_eg': 'arabic',
    'ar_sa': 'arabic',
    'bg_bg': 'bulgarian',
    'bn_bd': 'bengali',
    'bn_in': 'bengali',
    'ca_es': 'catalan',
    'cs_cz': 'czech',
    'da_dk': 'danish',
    'de_de': 'german',
    'el_gr': 'greek',
    'es_mx': 'spanish',
    'et_ee': 'estonian',
    'fa_ir': 'farsi',
    'fi_fi': 'finnish',
    'fil_ph': 'filipino',
    'fr_ca': 'french',
    'fr_fr': 'french',
    'gu_in': 'gujarati',
    'he_il': 'hebrew',
    'hi_in': 'hindi',
    'hr_hr': 'croatian',
    'hu_hu': 'hungarian',
    'id_id': 'indonesian',
    'is_is': 'icelandic',
    'it_it': 'italian',
    'ja_jp': 'japanese',
    'kn_in': 'kannada',
    'ko_kr': 'korean',
    'lt_lt': 'lithuanian',
    'lv_lv': 'latvian',
    'ml_in': 'malayalam',
    'mr_in': 'marathi',
    'nl_nl': 'dutch',
    'no_no': 'norwegian',
    'pa_in': 'punjabi',
    'pl_pl': 'polish',
    'pt_br': 'portuguese',
    'pt_pt': 'portuguese',
    'ro_ro': 'romanian',
    'ru_ru': 'russian',
    'sk_sk': 'slovak',
    'sl_si': 'slovenian',
    'sr_rs': 'serbian',
    'sv_se': 'swedish',
    'sw_ke': 'swahili',
    'sw_tz': 'swahili',
    'ta_in': 'tamil',
    'te_in': 'telugu',
    'th_th': 'thai',
    'tr_tr': 'turkish',
    'uk_ua': 'ukrainian',
    'ur_pk': 'urdu',
    'vi_vn': 'vietnamese',
    'zh_cn': 'mandarin',
    'zh_tw': 'mandarin',
    'zu_za': 'zulu',
    'en': 'english',
}


@register_benchmark(
    BenchmarkMeta(
        name='wmt24pp',
        pretty_name='WMT2024++',
        dataset_id='extraordinarylab/wmt24pp',
        tags=[Tags.MULTI_LINGUAL, Tags.MT],
        description=(
            'WMT2024 news translation benchmark supporting multiple language pairs. '
            'Each subset represents a specific translation direction'
        ),
        subset_list=LANGUAGE_PAIRS,
        eval_split='test',
        metric_list=[{
            'bleu': {}
        }, {
            'bert_score': {
                'model_id_or_path': 'AI-ModelScope/xlm-roberta-large',
                'model_type': 'xlm-roberta-large'
            }
        }, {
            'comet': {
                'model_id_or_path': 'evalscope/wmt22-comet-da',
            }
        }],
        few_shot_num=0,
        prompt_template=PROMPT_TEMPLATE,
    )
)
class WMT24PPAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs: Any) -> None:
        """Initialize adapter and configure dataset subsets."""
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.use_batch_scoring = True  # Enable batch scoring

        # Replace dict-style check with list[dict]-aware check
        if self.has_metric('comet'):
            check_import('comet', 'unbabel-comet', raise_error=True, feature_name='COMETScore Metric')

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.
        """
        source_text = str(record['source'])
        target_text = str(record['target'])
        language_pair = str(record['language_pair'])
        source_language, target_language = language_pair.split('-')

        # Format the generation prompt with the text
        input_prompt = self.prompt_template.format(
            source_text=source_text,
            source_language=LANGUAGE_BY_CODE[source_language],
            target_language=LANGUAGE_BY_CODE[target_language],
        )

        # Create content list for the input
        content_list = [ContentText(text=input_prompt)]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=target_text,
            subset_key=language_pair,
            metadata={
                'source_text': source_text,
                'target_text': target_text,
                'source_language': source_language,
                'target_language': target_language,
            },
        )

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Compute per-sample translation metrics."""
        # Create a Score object for the current sample
        score = Score(
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
            value={},
        )

        # ---- BLEU ----
        if self.has_metric('bleu'):
            try:
                from evalscope.metrics import bleu_ngram_one_sample

                bleu_results = bleu_ngram_one_sample(filtered_prediction, reference)
                score.value.update(bleu_results)
            except Exception as e:
                logger.warning(f'[WMT24PPAdapter] BLEU single-sample calculation failed: {e}')
        return score

    def batch_match_score(
        self,
        original_predictions: List[str],
        filtered_predictions: List[str],
        references: List[str],
        task_states: List[TaskState],
    ) -> List[Score]:
        """Compute batched translation metrics (BLEU, BERTScore, COMET)."""
        scores: List[Score] = []
        for i in range(len(original_predictions)):
            score = Score(
                extracted_prediction=filtered_predictions[i],
                prediction=original_predictions[i],
                value={},
            )
            scores.append(score)

        # ---- BLEU (per-sample within batch) ----
        if self.has_metric('bleu'):
            try:
                from evalscope.metrics import bleu_ngram_one_sample

                for i in range(len(scores)):
                    bleu_results = bleu_ngram_one_sample(filtered_predictions[i], references[i])
                    scores[i].value.update(bleu_results)
            except Exception as e:
                logger.warning(f'[WMT24PPAdapter] BLEU batch calculation failed: {e}')

        # ---- BERTScore ----
        if self.has_metric('bert_score'):
            try:
                from evalscope.metrics.metric import BertScore

                score_args = self.get_metric_args('bert_score')
                bert_scorer = BertScore(**score_args)
                bert_score_f1 = bert_scorer.apply(filtered_predictions, references)
                for i in range(len(scores)):
                    scores[i].value.update({'bert_score': bert_score_f1[i]})
            except Exception as e:
                logger.warning(f'[WMT24PPAdapter] BERTScore batch calculation failed: {e}')

        # ---- COMET ----
        if self.has_metric('comet'):
            try:
                from evalscope.metrics.metric import COMETScore

                score_args = self.get_metric_args('comet')
                comet_scorer = COMETScore(**score_args)
                data = [{
                    'src': st.metadata.get('source_text'),
                    'mt': pred,
                    'ref': ref
                } for pred, ref, st in zip(filtered_predictions, references, task_states)]
                comet_scores = comet_scorer.apply(data)
                for i in range(len(scores)):
                    scores[i].value.update({'comet': comet_scores[i]})
            except Exception as e:
                logger.warning(f'[WMT24PPAdapter] COMET batch calculation failed: {e}')

        return scores
