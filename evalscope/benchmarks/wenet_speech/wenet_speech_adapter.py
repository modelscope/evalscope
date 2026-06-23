# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import AudioLanguageAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='wenet_speech',
        pretty_name='WenetSpeech',
        dataset_id='lmms-lab/WenetSpeech',
        tags=[Tags.AUDIO, Tags.SPEECH_RECOGNITION],
        description="""
## Overview

WenetSpeech is a large-scale Mandarin Chinese speech corpus with over 10,000 hours of multi-domain transcribed audio data, designed for speech recognition research.

## Task Description

- **Task Type**: Automatic Speech Recognition (ASR)
- **Input**: Audio recordings with Mandarin Chinese speech
- **Output**: Transcribed text in Chinese
- **Domain**: Multi-domain (internet, meeting)

## Key Features

- Large-scale Mandarin Chinese speech corpus (10,000+ hours)
- Multi-domain coverage: internet content, meetings
- High-quality transcriptions
- Suitable for evaluating Chinese ASR systems
- Supports mixed Chinese-English text evaluation

## Evaluation Notes

- Default configuration uses **test_net** split
- Subsets by domain: **dev** (development), **test_meeting** (meeting domain), **test_net** (internet domain)
- Primary metric: **MER** (Mixed Error Rate)
- MER tokenizes Chinese characters individually and English words as whole tokens
- Prompt: "Please listen to the audio and transcribe what you hear"
""",
        subset_list=['dev', 'test_meeting', 'test_net'],
        eval_split='test_net',
        metric_list=['mer'],
        prompt_template='Please listen to the audio and transcribe what you hear. '
        'Please only provide the transcription without any additional commentary. '
        'Do not include any punctuation.',
    )
)
class WenetSpeechAdapter(AudioLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_as_subset = True

    def record_to_sample(self, record) -> Sample:
        content_list = [ContentText(text=self.prompt_template)]
        audio_bytes = self._to_wav(record['audio']['bytes'])
        audio_base64 = bytes_to_base64(audio_bytes, format='wav', add_header=True, content_type='audio')
        content_list.append(ContentAudio(audio=audio_base64, format='wav'))

        return Sample(
            input=[ChatMessageUser(content=content_list)], target=record['text'], metadata={
                'text': record['text'],
            }
        )
