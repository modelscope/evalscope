import os
import re
import requests
from typing import List, Optional

from evalscope.api.metric import Metric
from evalscope.api.registry import register_metric
from evalscope.utils.import_utils import check_import


@register_metric(name='wer')
class WER(Metric):

    def __init__(self, language: str = 'English'):
        self.language = language

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        from evalscope.metrics.utils.text_normalizer.wer import wer

        return [wer([ref], [pred], self.language) for pred, ref in zip(predictions, references)]


@register_metric(name='audio_wer')
class AudioWER(Metric):
    """WER metric for generated audio using a remote ASR endpoint."""

    DEFAULT_OPENAI_API_BASE = 'https://api.openai.com/v1'
    DEFAULT_SEED_TTS_TRANSCRIPTIONS_PATH = '/audio/transcriptions'
    DEFAULT_SEED_TTS_RESPONSES_PATH = '/responses'
    SUPPORTED_AUDIO_PROTOCOLS = {'transcriptions', 'responses'}

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        language: str = 'en',
        api_protocol: Optional[str] = None,
        prompt: Optional[str] = None,
        timeout: float = 60,
    ):
        self.api_base = (
            api_base or os.getenv('SEED_TTS_EVAL_ASR_API_BASE') or os.getenv('OPENAI_BASE_URL')
            or self.DEFAULT_OPENAI_API_BASE
        ).rstrip('/')
        self.api_key = api_key or os.getenv('SEED_TTS_EVAL_ASR_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.model = model or os.getenv('SEED_TTS_EVAL_ASR_MODEL') or 'whisper-1'
        self.language = self._normalize_language(language)
        self.api_protocol = self._resolve_protocol(api_protocol)
        self.prompt = prompt or os.getenv('SEED_TTS_EVAL_ASR_PROMPT') or 'Transcribe the speech. Return only the text.'
        self.timeout = timeout
        self.transcriptions: List[str] = []
        self._session = requests.Session()

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        from evalscope.metrics.utils.text_normalizer.wer import normalize_text, wer

        self.transcriptions = [self._transcribe(prediction) for prediction in predictions]
        scores = []
        for transcript, reference in zip(self.transcriptions, references):
            normalized_prediction = normalize_text(transcript, self.language)
            normalized_reference = normalize_text(reference, self.language)
            scores.append(wer([normalized_reference], [normalized_prediction], self.language))
        return scores

    def _transcribe(self, audio: str) -> str:
        if not self.api_key:
            raise ValueError('api_key is required for audio_wer. Set SEED_TTS_EVAL_ASR_API_KEY or OPENAI_API_KEY.')

        if self.api_protocol == 'responses':
            return self._transcribe_with_responses(audio)
        if self.api_protocol != 'transcriptions':
            raise ValueError(f'Unsupported audio_wer api_protocol: {self.api_protocol}')
        return self._transcribe_with_transcriptions(audio)

    def _transcribe_with_transcriptions(self, audio: str) -> str:
        from evalscope.utils.url_utils import file_as_data, is_data_uri

        endpoint = self._transcription_endpoint()
        audio_bytes, mime_type = file_as_data(audio, default_mime_type='audio/wav')
        filename = 'audio.wav' if is_data_uri(audio) else os.path.basename(audio.split('?', 1)[0]) or 'audio.wav'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        data = {
            'model': self.model,
            'response_format': 'json',
        }
        if self.language:
            data['language'] = 'zh' if self.language == 'cmn_hans' else self.language

        response = self._session.post(
            endpoint,
            headers=headers,
            files={'file': (filename, audio_bytes, mime_type)},
            data=data,
            timeout=self.timeout,
        )
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError:
            return response.text.strip()
        if not isinstance(payload, dict):
            raise ValueError(f'Unexpected response payload format (expected dict): {payload}')

        text = payload.get('text') or payload.get('transcription')
        if text is None and isinstance(payload.get('result'), dict):
            text = payload['result'].get('text')
        if text is None:
            raise ValueError(f'No transcription text found in response: {payload}')
        return str(text)

    def _transcribe_with_responses(self, audio: str) -> str:
        response = self._session.post(
            self._responses_endpoint(),
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': self.model,
                'input': [{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_audio',
                            'audio_url': self._audio_url(audio),
                        },
                        {
                            'type': 'input_text',
                            'text': self.prompt,
                        },
                    ],
                }],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f'Unexpected response payload format (expected dict): {payload}')

        output_text = payload.get('output_text')
        if output_text:
            return str(output_text)

        text_parts: List[str] = []
        for item in payload.get('output', []):
            if item.get('type') != 'message':
                continue
            for content in item.get('content', []):
                if content.get('type') in {'output_text', 'text'} and content.get('text'):
                    text_parts.append(str(content['text']))
        if not text_parts:
            raise ValueError(f'No transcription text found in response: {payload}')
        return '\n'.join(text_parts).strip()

    def _transcription_endpoint(self) -> str:
        return self._build_api_endpoint(self.api_base, self.DEFAULT_SEED_TTS_TRANSCRIPTIONS_PATH)

    def _responses_endpoint(self) -> str:
        return self._build_api_endpoint(self.api_base, self.DEFAULT_SEED_TTS_RESPONSES_PATH)

    @staticmethod
    def _audio_url(audio: str) -> str:
        from evalscope.utils.url_utils import file_as_data_uri, is_data_uri, is_http_url

        if is_data_uri(audio) or is_http_url(audio):
            return audio
        return file_as_data_uri(audio, default_mime_type='audio/wav')

    @staticmethod
    def _normalize_language(language: str) -> str:
        language_map = {
            'zh': 'cmn_hans',
            'cn': 'cmn_hans',
            'cmn': 'cmn_hans',
            'chinese': 'cmn_hans',
            'en': 'en',
            'english': 'en',
        }
        return language_map.get((language or '').lower(), language)

    @staticmethod
    def _build_api_endpoint(api_base: str, suffix: str) -> str:
        if api_base.endswith(suffix):
            return api_base
        return f'{api_base.rstrip("/")}/{suffix.lstrip("/")}'

    def _resolve_protocol(self, api_protocol: Optional[str]) -> str:
        protocol = (api_protocol or os.getenv('SEED_TTS_EVAL_ASR_API_PROTOCOL') or 'transcriptions').lower()
        if protocol not in self.SUPPORTED_AUDIO_PROTOCOLS:
            raise ValueError(f'Unsupported audio_wer api_protocol: {protocol}')
        return protocol


@register_metric(name='cer')
class CER(Metric):

    def __init__(self, language: str = 'en'):
        self.language = language

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        from jiwer import cer as jiwer_cer

        from evalscope.metrics.utils.text_normalizer.wer import normalize_text

        return [
            jiwer_cer(normalize_text(ref, self.language), normalize_text(pred, self.language))
            for pred, ref in zip(predictions, references)
        ]


@register_metric(name='mer')
class MER(Metric):
    """Mixed Error Rate for Chinese-English mixed text.

    Tokenizes Chinese characters individually and English words as whole tokens,
    then computes edit distance / max(len_ref, len_hyp).
    """

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        import editdistance

        scores = []
        for pred, ref in zip(predictions, references):
            ref_tokens = self._tokenize(ref)
            hyp_tokens = self._tokenize(pred)
            distance = editdistance.eval(ref_tokens, hyp_tokens)
            max_len = max(len(ref_tokens), len(hyp_tokens))
            scores.append(distance / max_len if max_len > 0 else 0.0)
        return scores

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize: Chinese characters as single tokens, English words as whole tokens."""
        tokens: List[str] = []
        i = 0
        while i < len(text):
            char = text[i]
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
                i += 1
            else:
                match = re.match(r"[a-zA-Z']+\w*", text[i:])
                if match:
                    tokens.append(match.group(0))
                    i += match.end()
                else:
                    i += 1
        return tokens
