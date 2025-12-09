# Copyright (c) Alibaba, Inc. and its affiliates.

from collections import OrderedDict

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentAudio, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

FLEURS_LANG_TO_ID = OrderedDict([
    ('Afrikaans', 'af'),
    ('Amharic', 'am'),
    ('Arabic', 'ar'),
    ('Armenian', 'hy'),
    ('Assamese', 'as'),
    ('Asturian', 'ast'),
    ('Azerbaijani', 'az'),
    ('Belarusian', 'be'),
    ('Bengali', 'bn'),
    ('Bosnian', 'bs'),
    ('Bulgarian', 'bg'),
    ('Burmese', 'my'),
    ('Catalan', 'ca'),
    ('Cebuano', 'ceb'),
    ('Mandarin Chinese', 'cmn_hans'),
    ('Cantonese Chinese', 'yue_hant'),
    ('Croatian', 'hr'),
    ('Czech', 'cs'),
    ('Danish', 'da'),
    ('Dutch', 'nl'),
    ('English', 'en'),
    ('Estonian', 'et'),
    ('Filipino', 'fil'),
    ('Finnish', 'fi'),
    ('French', 'fr'),
    ('Fula', 'ff'),
    ('Galician', 'gl'),
    ('Ganda', 'lg'),
    ('Georgian', 'ka'),
    ('German', 'de'),
    ('Greek', 'el'),
    ('Gujarati', 'gu'),
    ('Hausa', 'ha'),
    ('Hebrew', 'he'),
    ('Hindi', 'hi'),
    ('Hungarian', 'hu'),
    ('Icelandic', 'is'),
    ('Igbo', 'ig'),
    ('Indonesian', 'id'),
    ('Irish', 'ga'),
    ('Italian', 'it'),
    ('Japanese', 'ja'),
    ('Javanese', 'jv'),
    ('Kabuverdianu', 'kea'),
    ('Kamba', 'kam'),
    ('Kannada', 'kn'),
    ('Kazakh', 'kk'),
    ('Khmer', 'km'),
    ('Korean', 'ko'),
    ('Kyrgyz', 'ky'),
    ('Lao', 'lo'),
    ('Latvian', 'lv'),
    ('Lingala', 'ln'),
    ('Lithuanian', 'lt'),
    ('Luo', 'luo'),
    ('Luxembourgish', 'lb'),
    ('Macedonian', 'mk'),
    ('Malay', 'ms'),
    ('Malayalam', 'ml'),
    ('Maltese', 'mt'),
    ('Maori', 'mi'),
    ('Marathi', 'mr'),
    ('Mongolian', 'mn'),
    ('Nepali', 'ne'),
    ('Northern-Sotho', 'nso'),
    ('Norwegian', 'nb'),
    ('Nyanja', 'ny'),
    ('Occitan', 'oc'),
    ('Oriya', 'or'),
    ('Oromo', 'om'),
    ('Pashto', 'ps'),
    ('Persian', 'fa'),
    ('Polish', 'pl'),
    ('Portuguese', 'pt'),
    ('Punjabi', 'pa'),
    ('Romanian', 'ro'),
    ('Russian', 'ru'),
    ('Serbian', 'sr'),
    ('Shona', 'sn'),
    ('Sindhi', 'sd'),
    ('Slovak', 'sk'),
    ('Slovenian', 'sl'),
    ('Somali', 'so'),
    ('Sorani-Kurdish', 'ckb'),
    ('Spanish', 'es'),
    ('Swahili', 'sw'),
    ('Swedish', 'sv'),
    ('Tajik', 'tg'),
    ('Tamil', 'ta'),
    ('Telugu', 'te'),
    ('Thai', 'th'),
    ('Turkish', 'tr'),
    ('Ukrainian', 'uk'),
    ('Umbundu', 'umb'),
    ('Urdu', 'ur'),
    ('Uzbek', 'uz'),
    ('Vietnamese', 'vi'),
    ('Welsh', 'cy'),
    ('Wolof', 'wo'),
    ('Xhosa', 'xh'),
    ('Yoruba', 'yo'),
    ('Zulu', 'zu'),
])


@register_benchmark(
    BenchmarkMeta(
        name='fleurs',
        pretty_name='FLEURS',
        dataset_id='lmms-lab/fleurs',
        tags=[Tags.AUDIO, Tags.MULTI_LINGUAL, Tags.SPEECH_RECOGNITION],
        description=
        'FLEURS is a massively multilingual benchmark with 102 languages for evaluating ASR, spoken language understanding, and speech translation',  # noqa: E501
        subset_list=['cmn_hans_cn', 'en_us', 'yue_hant_hk'],
        eval_split='test',
        metric_list=['wer'],
        prompt_template='Please recognize the speech and only output the recognized content:',
    )
)
class FLEURSAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        content_list = [ContentText(text=self.prompt_template)]
        audio_base64 = bytes_to_base64(record['audio']['bytes'], format='wav', add_header=True, content_type='audio')
        content_list.append(ContentAudio(audio=audio_base64, format='wav'))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['transcription'],
            metadata={
                'id': record['id'],
                'num_samples': record['num_samples'],
                'raw_transcription': record['raw_transcription'],
                'language': record['language'],
                'gender': record['gender'],
                'lang_id': FLEURS_LANG_TO_ID[record['language']],
                'lang_group_id': record['lang_group_id'],
            }
        )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        from evalscope.metrics.text_normalizer.wer import normalize_text, wer

        language = task_state.metadata['lang_id']

        normalized_prediction = normalize_text(original_prediction, language)
        normalized_reference = normalize_text(reference, language)
        score = Score(
            extracted_prediction=normalized_prediction,
            prediction=original_prediction,
        )

        wer_score = wer([normalized_reference], [normalized_prediction], language)
        score.value = {'wer': wer_score}
        return score
