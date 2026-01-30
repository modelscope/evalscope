# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentAudio, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='librispeech',
        pretty_name='LibriSpeech',
        dataset_id='lmms-lab/Librispeech-concat',
        tags=[Tags.AUDIO, Tags.SPEECH_RECOGNITION],
        description="""
## Overview

LibriSpeech is a large-scale corpus of approximately 1,000 hours of read English speech derived from audiobooks. It is one of the most widely used benchmarks for evaluating automatic speech recognition (ASR) systems.

## Task Description

- **Task Type**: Automatic Speech Recognition (ASR)
- **Input**: Audio recordings of read English speech from audiobooks
- **Output**: Transcribed text
- **Language**: English

## Key Features

- 1,000 hours of high-quality read speech
- Derived from LibriVox audiobooks (public domain)
- Clean and "other" test sets for varying difficulty
- Widely used baseline for ASR research
- Standardized evaluation protocol

## Evaluation Notes

- Default configuration uses **test_clean** split
- Primary metric: **Word Error Rate (WER)**
- Text normalization applied during evaluation
- Prompt: "Please recognize the speech and only output the recognized content"
- Metadata includes audio ID and duration information
""",  # noqa: E501
        eval_split='test_clean',
        metric_list=['wer'],
        prompt_template='Please recognize the speech and only output the recognized content:',
    )
)
class LibriSpeechAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        content_list = [ContentText(text=self.prompt_template)]
        audio_base64 = bytes_to_base64(record['audio']['bytes'], format='wav', add_header=True, content_type='audio')
        content_list.append(ContentAudio(audio=audio_base64, format='wav'))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['transcript'],
            metadata={
                'audio_id': record['audio_id'],
                'audio_duration': record['audio_duration'],
            }
        )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        from evalscope.metrics.text_normalizer.wer import normalize_text, wer

        language = 'en'

        normalized_prediction = normalize_text(original_prediction, language)
        normalized_reference = normalize_text(reference, language)
        score = Score(
            extracted_prediction=normalized_prediction,
            prediction=original_prediction,
        )

        wer_score = wer([normalized_reference], [normalized_prediction], language)
        score.value = {'wer': wer_score}
        return score
