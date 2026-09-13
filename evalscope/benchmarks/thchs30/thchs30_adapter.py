# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import AudioLanguageAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64


@register_benchmark(
    BenchmarkMeta(
        name='thchs30',
        pretty_name='THCHS-30',
        dataset_id='evalscope/THCHS-30',
        tags=[Tags.AUDIO, Tags.CHINESE, Tags.SPEECH_RECOGNITION],
        description="""
## Overview

THCHS-30 is a Mandarin Chinese read-speech corpus with phone-level time alignments. This EvalScope benchmark evaluates phoneme recognition from speech using the aligned IPA phone sequences released with the dataset.

## Task Description

- **Task Type**: Phoneme Recognition
- **Input**: A 16 kHz Mandarin Chinese speech recording
- **Output**: A space-separated sequence of IPA phone tokens with tone diacritics
- **Domain**: Read Mandarin Chinese speech

## Key Features

- 13,388 utterances split into 10,000 train, 893 validation, and 2,495 test samples
- Phone-level start and end timestamps, including `[SIL]` silence intervals
- Original Hanzi transcripts, speaker identifiers, durations, and split labels retained as sample metadata
- Test evaluation uses the release's IPA phone inventory and tone annotations

## Evaluation Notes

- Default configuration evaluates the `test` split only; no few-shot examples are used
- Primary metric: Phone Error Rate (PER), computed as phone-token edit distance divided by reference phone count
- The model must output only IPA phone tokens separated by spaces; timestamps are metadata for downstream analysis, not model targets
- Audio is passed as WAV data through EvalScope's standard audio message format
""",
        paper_url='https://arxiv.org/abs/1512.01882',
        eval_split='test',
        few_shot_mode='disabled',
        metric_list=['per'],
        prompt_template='Transcribe the audio as IPA phone tokens. Output only tokens separated by single spaces.',
        evaluation_version='v1.0',
    )
)
class THCHS30Adapter(AudioLanguageAdapter):
    """Adapt the aligned THCHS-30 release for phoneme-recognition evaluation."""

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert one aligned audio record into an EvalScope sample."""
        audio_base64 = bytes_to_base64(record['audio']['bytes'], format='wav', add_header=True, content_type='audio')
        metadata_keys = (
            'utt_id',
            'text',
            'phones',
            'phone_starts',
            'phone_ends',
            'language',
            'speaker_id',
            'duration',
            'split',
        )

        return Sample(
            input=[
                ChatMessageUser(
                    content=[
                        ContentText(text=self.prompt_template),
                        ContentAudio(audio=audio_base64, format='wav'),
                    ]
                )
            ],
            target=' '.join(record['phones']),
            metadata={key: record[key] for key in metadata_keys if key in record},
        )
