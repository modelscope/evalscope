# Copyright (c) Alibaba, Inc. and its affiliates.

import base64
import os
import shutil
from typing import Any, Dict, List, Optional, Union

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import get_metric, register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64, safe_filename
from evalscope.utils.logger import get_logger
from evalscope.utils.url_utils import data_uri_to_base64, is_data_uri, is_http_url

logger = get_logger()

PROMPT_TEMPLATE = (
    'Use the reference audio and prompt transcript to synthesize the target text in the same speaker voice.\n'
    'Prompt transcript: {prompt_text}\n'
    'Target text: {text}\n'
    'Return only the generated audio.'
)


@register_benchmark(
    BenchmarkMeta(
        name='seed_tts_eval',
        pretty_name='Seed-TTS-Eval',
        dataset_id='evalscope/Seed-TTS-Eval',
        tags=[Tags.AUDIO, Tags.TEXT_TO_SPEECH],
        description="""
## Overview

Seed-TTS-Eval is an objective benchmark for zero-shot text-to-speech and voice conversion evaluation. It uses out-of-domain English and Mandarin samples from Common Voice and DiDiSpeech-2, and the official evaluation focuses on intelligibility and speaker consistency.

## Task Description

- **Task Type**: Zero-shot text-to-speech generation
- **Input**: Reference speaker audio, prompt transcript, and target text
- **Output**: Synthesized speech audio for the target text using the reference speaker
- **Languages**: English and Mandarin

## Evaluation Notes

- Default subsets: **en** and **zh**
- The evaluated TTS model must return generated audio as `ContentAudio`, or return an audio path, URL, or data URI as the completion text
- EvalScope provides `eval_type="text2speech"` for HTTP TTS services.
  - Volcengine provider: configure `model="seed-tts-2.0"`, `api_url="https://openspeech.bytedance.com/api/v3/tts/unidirectional"`, and `model_args={"speaker": "..."}`
  - OpenAI provider: configure `model="tts-1"` (or `tts-1-hd`), `api_url="https://api.openai.com/v1"`, and `model_args={"provider": "openai", "voice": "nova"}`
- Default metric: **audio_wer**, which transcribes generated audio through an OpenAI-compatible `/audio/transcriptions` endpoint and computes WER/CER-style error rate with language-specific normalization
- Configure the ASR endpoint via `metric_list`, or set `SEED_TTS_EVAL_ASR_API_BASE`, `SEED_TTS_EVAL_ASR_API_KEY`, `SEED_TTS_EVAL_ASR_MODEL`, and `SEED_TTS_EVAL_ASR_API_PROTOCOL`
- For Volcengine Ark audio-understanding models, set `api_protocol="responses"` and use a model that supports audio input, such as `doubao-seed-2-0-lite-260428`
- Speaker similarity is part of the official benchmark, but it requires a separate speaker verification backend and is not enabled by default
""",  # noqa: E501
        paper_url='https://arxiv.org/abs/2406.02430',
        subset_list=['en', 'zh'],
        default_subset='en',
        eval_split='train',
        few_shot_num=0,
        metric_list=['audio_wer'],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class SeedTTSEvalAdapter(DefaultDataAdapter):
    """Adapter for Seed-TTS-Eval zero-shot text-to-speech evaluation."""

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompt_text = str(record['prompt_text'])
        target_text = str(record['text'])
        prompt_audio = self._prompt_audio(record)
        language = self._language_from_subset()

        content_list = [
            ContentAudio(audio=prompt_audio, format='wav'),
            ContentText(text=self.prompt_template.format(prompt_text=prompt_text, text=target_text)),
        ]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=target_text,
            subset_key=self.current_subset_name or language,
            metadata={
                'filename': record.get('filename'),
                'prompt_text': prompt_text,
                'text': target_text,
                'prompt_audio_path': record.get('WavPath'),
                'reference_audio_path': record.get('ans'),
                'language': language,
                'wer_language': 'cmn_hans' if language == 'zh' else 'en',
            },
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        return self.prompt_template.format(
            prompt_text=sample.metadata.get('prompt_text', ''),
            text=sample.metadata.get('text', sample.target),
        )

    def _on_inference_end(
        self, model: Model, sample: Sample, model_output: ModelOutput, output_dir: str, **kwargs: Any
    ) -> TaskState:
        task_state = super()._on_inference_end(model, sample, model_output, output_dir, **kwargs)
        generated_audio = self._save_generated_audio(model_output, sample, output_dir)
        if generated_audio:
            task_state.metadata['generated_audio_path'] = generated_audio
            model_output.completion = generated_audio
        return task_state

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        generated_audio = task_state.metadata.get('generated_audio_path') or filtered_prediction or original_prediction
        score = Score(
            extracted_prediction=generated_audio,
            prediction=generated_audio,
        )

        for metric in self.metric_list:
            metric_name = self._metric_name(metric)
            try:
                metric_args = self.get_metric_args(metric_name)
                if metric_name == 'audio_wer':
                    metric_args.setdefault('language', task_state.metadata.get('wer_language', 'en'))
                metric_cls = get_metric(metric_name)
                metric_func = metric_cls(**metric_args)
                metric_score = metric_func(generated_audio, reference)
                score.value[metric_name] = metric_score
                transcriptions = getattr(metric_func, 'transcriptions', None)
                if metric_name == 'audio_wer' and transcriptions:
                    score.metadata['transcription'] = transcriptions[0]
            except Exception as e:
                logger.error(f'Error calculating metric {metric}: {e}')
                score.value[metric_name] = 0
                score.metadata[metric_name] = f'error: {str(e)}'

        return score

    def _prompt_audio(self, record: Dict[str, Any]) -> str:
        audio = record.get('audio') or {}
        audio_bytes = audio.get('bytes') if isinstance(audio, dict) else None
        if audio_bytes:
            return bytes_to_base64(audio_bytes, format='wav', add_header=True, content_type='audio')
        if isinstance(audio, dict) and audio.get('path'):
            return str(audio['path'])
        if record.get('WavPath'):
            return str(record['WavPath'])
        raise ValueError('Seed-TTS-Eval record must contain prompt audio.')

    def _save_generated_audio(self, model_output: ModelOutput, sample: Sample, output_dir: str) -> Optional[str]:
        if model_output.empty:
            return None
        audio_content = self._first_audio_content(model_output)
        if audio_content is None:
            return None

        audio_id = safe_filename(str(sample.metadata.get('filename') or sample.id))
        audio_format = audio_content.format
        output_path = os.path.join(
            output_dir,
            'audios',
            'seed_tts_eval',
            f'{audio_id}_{sample.group_id}.{audio_format}',
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.isfile(audio_content.audio):
            shutil.copyfile(audio_content.audio, output_path)
        elif is_http_url(audio_content.audio):
            return audio_content.audio
        else:
            audio_base64 = audio_content.audio
            if is_data_uri(audio_base64):
                audio_base64 = data_uri_to_base64(audio_base64)
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(audio_base64))
        return output_path

    @staticmethod
    def _first_audio_content(model_output: ModelOutput) -> Optional[ContentAudio]:
        content = model_output.message.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, ContentAudio):
                    return item
        return None

    @staticmethod
    def _metric_name(metric: Union[str, Dict[str, Any]]) -> str:
        if isinstance(metric, str):
            return metric
        return list(metric.keys())[0]

    def _language_from_subset(self) -> str:
        if self.current_subset_name in {'en', 'zh'}:
            return self.current_subset_name
        return self.default_subset
