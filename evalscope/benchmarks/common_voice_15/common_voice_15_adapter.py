# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import AudioLanguageAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

# Mapping from dataset locale codes to normalizer language codes
LOCALE_TO_LANG = {
    'en': 'en',
    'zh-CN': 'cmn_hans',
    'fr': 'fr',
}


@register_benchmark(
    BenchmarkMeta(
        name='common_voice_15',
        pretty_name='CommonVoice15',
        dataset_id='lmms-lab/common_voice_15',
        tags=[Tags.AUDIO, Tags.MULTI_LINGUAL, Tags.SPEECH_RECOGNITION],
        description="""
## Overview

Common Voice 15 is a massively multilingual speech corpus collected by Mozilla, covering 114 languages with thousands of hours of validated speech data from volunteers worldwide.

## Task Description

- **Task Type**: Automatic Speech Recognition (ASR)
- **Input**: Audio recordings with speech in various languages
- **Output**: Transcribed text in the corresponding language
- **Languages**: 114 languages including English, Mandarin Chinese, French, and many more

## Key Features

- Crowd-sourced speech recordings with community validation
- Diverse speaker demographics (age, gender, accent)
- Multiple languages with varying amounts of data
- CC-0 licensed for open research and commercial use
- High-quality transcriptions validated by multiple listeners

## Evaluation Notes

- Default configuration uses **test** split
- Primary metric: **Word Error Rate (WER)**
- Default subsets: `en` (English), `zh-CN` (Mandarin Chinese), `fr` (French)
- Language-specific text normalization applied during evaluation
- Prompt: "Please recognize the speech and only output the recognized content"
""",
        subset_list=['en', 'zh-CN', 'fr'],
        eval_split='test',
        metric_list=['wer'],
        prompt_template='Please recognize the speech and only output the recognized content:',
    )
)
class CommonVoice15Adapter(AudioLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        content_list = [ContentText(text=self.prompt_template)]
        audio_base64 = bytes_to_base64(record['audio']['bytes'], format='mp3', add_header=True, content_type='audio')
        content_list.append(ContentAudio(audio=audio_base64, format='mp3'))

        locale = record.get('locale', 'en')

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['sentence'],
            subset_key=locale,
            metadata={
                'locale': locale,
                'path': record.get('path', ''),
                'lang_id': LOCALE_TO_LANG.get(locale, locale),
            }
        )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        from evalscope.metrics.utils.text_normalizer.wer import normalize_text, wer

        locale = task_state.metadata.get('locale', 'en')
        language = LOCALE_TO_LANG.get(locale, locale)

        normalized_prediction = normalize_text(filtered_prediction, language)
        normalized_reference = normalize_text(reference, language)
        score = Score(
            extracted_prediction=normalized_prediction,
            prediction=original_prediction,
        )

        wer_score = wer([normalized_reference], [normalized_prediction], language)
        score.value = {'wer': wer_score}
        return score
