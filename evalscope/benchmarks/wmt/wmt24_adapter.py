import os
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
}


@register_benchmark(
    BenchmarkMeta(
        name='wmt24pp',
        pretty_name='WMT2024++',
        dataset_id='extraordinarylab/wmt24pp',
        tags=[Tags.MULTI_LINGUAL],
        description=(
            'WMT2024 news translation benchmark supporting multiple language pairs. '
            'Each subset represents a specific translation direction'
        ),
        subset_list=LANGUAGE_PAIRS,
        eval_split='test',
        metric_list=['bleu', 'bert_score', 'comet'],
        few_shot_num=0,
        prompt_template=PROMPT_TEMPLATE,
    )
)
class WMT24PPAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        """Initialize adapter and configure dataset subsets."""
        super().__init__(**kwargs)
        self.reformat_subset = True

        self._load_nltk_resources()
        self._init_bert_scorer()
        self._init_comet_scorer()

    def _init_bert_scorer(self):
        check_import('bert_score', 'bert_score', raise_error=True, feature_name='Text similarity metrics')
        from bert_score import BERTScorer
        self.bert_scorer = BERTScorer(
            model_type='xlm-roberta-large',
            rescale_with_baseline=False,
        )

    def _init_comet_scorer(self):
        check_import('comet', 'unbabel-comet', raise_error=True, feature_name='Text similarity metrics')
        from comet import download_model, load_from_checkpoint
        model_path = download_model('Unbabel/wmt22-comet-da')
        self.comet_scorer = load_from_checkpoint(model_path)

    def _load_nltk_resources(self):
        check_import('nltk', 'nltk', raise_error=True, feature_name='NLTK')
        import nltk

        required_resources = ['punkt_tab', 'punkt', 'averaged_perceptron_tagger']

        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.
        """
        source_text = str(record['source'])
        target_text = str(record['target'])
        language_pair = str(record['language_pair'])
        source_language, target_language = language_pair.split('-')

        # Format the generation prompt with the text
        input_prompt = PROMPT_TEMPLATE.format(
            source_text=source_text,
            source_language=source_language.capitalize(),
            target_language=target_language.capitalize(),
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
            }
        )

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Compute translation metrics in batch mode, ensuring dataset consistency.

        This method supports efficient batched evaluation for BLEU, BERTScore, and COMET.
        It ensures proper ordering, memory control, and metric consistency across batches.

        Args:
            original_prediction: Raw model output.
            filtered_prediction: Cleaned or trimmed model output.
            reference: Ground truth translation.
            task_state: Task runtime state (used to detect final batch).

        Returns:
            A `Score` object containing computed metric results for the current sample.
        """
        # Initialize cache structures (only on first run)
        if not hasattr(self, '_batch_cache'):
            self._batch_cache = []
        if not hasattr(self, '_pending_scores'):
            self._pending_scores = []

        # Return any pending results first
        if self._pending_scores:
            return self._pending_scores.pop(0)

        # Create a Score object for the current sample
        score_obj = Score(prediction=original_prediction, extracted_prediction=filtered_prediction, value={})

        # Add current sample to batch cache
        self._batch_cache.append((original_prediction, filtered_prediction, reference, task_state, score_obj))

        # Check batch execution condition
        batch_limit = 128
        is_last_batch = getattr(task_state, 'is_last', False)
        should_compute = len(self._batch_cache) >= batch_limit or is_last_batch

        # Defer computation if batch is not yet full
        if not should_compute:
            return score_obj

        # Extract batch data
        preds = [x[1] for x in self._batch_cache]
        refs = [x[2] for x in self._batch_cache]
        states = [x[3] for x in self._batch_cache]
        score_objs = [x[4] for x in self._batch_cache]

        # ---- BLEU ----
        if 'bleu' in self.metric_list:
            try:
                from evalscope.metrics import bleu_ngram_one_sample
                bleu_results = [bleu_ngram_one_sample(p, r) for p, r in zip(preds, refs)]
                for s, b in zip(score_objs, bleu_results):
                    s.value.update(b)
            except Exception as e:
                logger.warning(f'[WMT24PPAdapter] BLEU batch calculation failed: {e}')

        # ---- BERTScore ----
        if 'bert_score' in self.metric_list:
            try:
                P, R, F1 = self.bert_scorer.score(preds, refs)
                for s, p, r, f1 in zip(score_objs, P, R, F1):
                    s.value.update({
                        'bertscore-precision': round(p.item(), 6),
                        'bertscore-recall': round(r.item(), 6),
                        'bertscore-f1': round(f1.item(), 6)
                    })
            except Exception as e:
                logger.warning(f'[WMT24PPAdapter] BERTScore batch failed: {e}')

        # ---- COMET ----
        if 'comet' in self.metric_list:
            try:
                data = [{
                    'src': st.metadata.get('source_text'),
                    'mt': pred,
                    'ref': ref
                } for pred, ref, st in zip(preds, refs, states)]
                model_output = self.comet_scorer.predict(data, batch_size=32, gpus=1, progress_bar=False)
                scores = model_output.scores if hasattr(model_output,
                                                        'scores') else [model_output.system_score] * len(data)
                for s, comet_val in zip(score_objs, scores):
                    s.value.update({'comet': round(float(comet_val), 6)})
            except Exception as e:
                logger.warning(f'[WMT24PPAdapter] COMET batch failed: {e}')

        # Determine main score name (priority order)
        for s in score_objs:
            if 'comet' in s.value:
                s.main_score_name = 'comet'
            elif 'bleu' in s.value:
                s.main_score_name = 'bleu'
            elif 'bertscore-f1' in s.value:
                s.main_score_name = 'bertscore-f1'

        # Queue computed results and clear batch cache
        self._pending_scores.extend(score_objs)
        self._batch_cache.clear()

        # Return the first result (EvalScope framework consumes one at a time)
        return self._pending_scores.pop(0)
