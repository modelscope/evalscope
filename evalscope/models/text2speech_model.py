import base64
import json
import os
import re
import requests
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from evalscope.api.messages import ChatMessage, ContentAudio
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils.io_utils import bytes_to_base64


class Text2SpeechAPI(ModelAPI):
    """Text-to-speech model provider."""

    VOLCENGINE_TTS_API_BASE = 'https://openspeech.bytedance.com/api/v3/tts'
    OPENAI_TTS_API_BASE = 'https://api.openai.com/v1'
    VOLCENGINE_FORMATS = {'mp3', 'wav'}
    OPENAI_FORMATS = {'mp3', 'wav', 'flac', 'opus', 'pcm', 'aac', 'm4a'}
    _TARGET_TEXT_PATTERN = re.compile(r'Target text:\s*(.*?)(?:\n\s*Return only\b|\Z)', re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        self.provider = (model_args.pop('provider', 'volcengine') or 'volcengine').lower()
        self.base_url = (base_url or '').rstrip('/')
        self.timeout = model_args.pop('timeout', 60)
        self.audio_format = model_args.pop('format', 'mp3')
        self.session = requests.Session()
        self._provider_kwargs = {}

        if self.provider == 'volcengine':
            self._init_volcengine(model_name=model_name, api_key=api_key, model_args=model_args)
        elif self.provider == 'openai':
            self._init_openai(model_name=model_name, api_key=api_key, model_args=model_args)
        else:
            raise ValueError(f'Unsupported text2speech provider: {self.provider}')

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        start_time = time.monotonic()
        if self.provider == 'volcengine':
            audio_bytes, metadata = self._generate_volcengine_audio(input, config.timeout or self.timeout)
        else:
            audio_bytes, metadata = self._generate_openai_audio(input, config.timeout or self.timeout)

        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice.from_content([
                    ContentAudio(
                        audio=bytes_to_base64(
                            audio_bytes,
                            format=self.audio_format,
                            add_header=True,
                            content_type='audio',
                        ),
                        format=self.audio_format,
                    )
                ])
            ],
            time=time.monotonic() - start_time,
            metadata=metadata,
        )

    def _init_volcengine(self, model_name: str, api_key: Optional[str], model_args: Dict[str, Any]) -> None:
        self.url = self._resolve_endpoint(
            base_url=self.base_url,
            default_base=self.VOLCENGINE_TTS_API_BASE,
            endpoint='unidirectional',
        )
        self.api_key = self._resolve_volcengine_api_key(api_key)
        self.resource_id = model_args.pop('resource_id', model_name)
        self.speaker = model_args.pop('speaker', None)
        self.user_id = model_args.pop('user_id', 'evalscope')
        self.audio_params = model_args.pop('audio_params', {})
        self.req_params = model_args.pop('req_params', {})
        self.app_id = model_args.pop('app_id', None)
        self.access_key = model_args.pop('access_key', None)
        self._provider_kwargs = model_args

        if not self.api_key and not (self.app_id and self.access_key):
            raise ValueError(
                'api_key is required for Volcengine text2speech. '
                'Set --api-key or VOLCENGINE_TTS_API_KEY.'
            )
        if not self.speaker and 'speaker' not in self.req_params:
            raise ValueError('model_args.speaker is required for Volcengine text2speech.')
        if self.audio_format not in self.VOLCENGINE_FORMATS:
            raise ValueError('text2speech output format must be "mp3" or "wav" for Volcengine.')

    def _init_openai(self, model_name: str, api_key: Optional[str], model_args: Dict[str, Any]) -> None:
        self.url = self._build_openai_tts_url(self.base_url or self.OPENAI_TTS_API_BASE)
        self.api_key = self._resolve_openai_api_key(api_key)
        self.voice = model_args.pop('voice', 'alloy')
        self.speed = model_args.pop('speed', None)
        self.extra_body = model_args.pop('extra_body', {})
        self._provider_kwargs = model_args

        if not self.api_key:
            raise ValueError('api_key is required for OpenAI text2speech. Set --api-key or OPENAI_API_KEY.')
        if self.audio_format not in self.OPENAI_FORMATS:
            raise ValueError(
                'text2speech output format must be one of '
                f'{", ".join(sorted(self.OPENAI_FORMATS))} for OpenAI.'
            )
        if self.speed is not None:
            try:
                self.speed = float(self.speed)
            except (TypeError, ValueError) as ex:
                raise ValueError('model_args.speed must be a number for OpenAI TTS.') from ex
            if self.speed < 0.25 or self.speed > 4.0:
                raise ValueError('model_args.speed must be in [0.25, 4.0] for OpenAI TTS.')

    @staticmethod
    def _build_openai_tts_url(base_url: str) -> str:
        base = base_url.rstrip('/')
        if base.endswith('/audio/speech'):
            return base
        return f'{base}/audio/speech'

    @staticmethod
    def _resolve_endpoint(base_url: str, default_base: str, endpoint: str) -> str:
        base = (base_url or default_base).rstrip('/')
        suffix = endpoint.strip('/')
        if base.endswith(f'/{suffix}'):
            return base
        return f'{base}/{endpoint}'

    @staticmethod
    def _resolve_openai_api_key(api_key: Optional[str]) -> Optional[str]:
        if api_key and api_key != 'EMPTY':
            return api_key
        return os.getenv('OPENAI_API_KEY') or os.getenv('EVALSCOPE_API_KEY')

    @staticmethod
    def _resolve_volcengine_api_key(api_key: Optional[str]) -> Optional[str]:
        if api_key and api_key != 'EMPTY':
            return api_key
        return os.getenv('VOLCENGINE_TTS_API_KEY') or os.getenv('VOLC_TTS_API_KEY')

    def _generate_volcengine_audio(self, input: List[ChatMessage], timeout: float) -> tuple[bytes, Dict[str, Any]]:
        payload = self._volcengine_payload(input)
        response = self.session.post(
            self.url,
            headers=self._volcengine_headers(),
            json=payload,
            stream=True,
            timeout=timeout,
        )
        response.raise_for_status()
        return self._collect_volcengine_audio(response)

    def _generate_openai_audio(self, input: List[ChatMessage], timeout: float) -> tuple[bytes, Dict[str, Any]]:
        payload = self._openai_payload(input)
        response = self.session.post(
            self.url,
            headers=self._openai_headers(),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return self._collect_openai_audio(response)

    def _volcengine_payload(self, input: List[ChatMessage]) -> Dict[str, Any]:
        audio_params = {
            'format': self.audio_format,
            **self.audio_params,
        }
        req_params = {
            'text': self._extract_tts_text(input),
            'audio_params': audio_params,
            **self.req_params,
        }
        if self.speaker:
            req_params.setdefault('speaker', self.speaker)

        return {
            'user': {
                'uid': self.user_id,
            },
            'req_params': req_params,
        }

    def _openai_payload(self, input: List[ChatMessage]) -> Dict[str, Any]:
        payload = {
            'model': self.model_name,
            'input': self._extract_tts_text(input),
            'voice': self.voice,
            'response_format': self.audio_format,
        }
        if self.speed is not None:
            payload['speed'] = self.speed
        payload.update(self._provider_kwargs)
        payload.update(self.extra_body)
        return payload

    def _volcengine_headers(self) -> Dict[str, str]:
        headers = {
            'Content-Type': 'application/json',
            'X-Api-Resource-Id': self.resource_id,
            'X-Api-Request-Id': str(uuid.uuid4()),
        }
        if self.api_key:
            headers['X-Api-Key'] = self.api_key
        else:
            headers['X-Api-App-Id'] = self.app_id
            headers['X-Api-Access-Key'] = self.access_key
        return headers

    def _openai_headers(self) -> Dict[str, str]:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

    def _extract_tts_text(self, input: List[ChatMessage]) -> str:
        text = '\n'.join(message.text for message in input if message.text).strip()
        match = self._TARGET_TEXT_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return text

    def _collect_volcengine_audio(self, response: requests.Response) -> tuple[bytes, Dict[str, Any]]:
        audio_chunks: List[bytes] = []
        metadata: Dict[str, Any] = {
            'provider': self.provider,
            'resource_id': self.resource_id,
            'request_id': response.request.headers.get('X-Api-Request-Id') if response.request else None,
            'log_id': response.headers.get('X-Tt-Logid') or response.headers.get('x-tt-logid'),
        }

        for event in self._decode_volcengine_events(response):
            code = event.get('code')
            if code not in (None, 0, 20000000):
                raise RuntimeError(f'Volcengine TTS error: code={code}, message={event.get("message")}')
            data = event.get('data')
            if isinstance(data, str) and data:
                audio_chunks.append(base64.b64decode(data))
            if code == 20000000:
                metadata['usage'] = event.get('usage')

        if not audio_chunks:
            raise ValueError('No audio data found in Volcengine TTS response.')
        return b''.join(audio_chunks), metadata

    @staticmethod
    def _collect_openai_audio(response: requests.Response) -> tuple[bytes, Dict[str, Any]]:
        audio_bytes = response.content
        if not audio_bytes:
            raise ValueError('No audio data found in OpenAI TTS response.')

        return audio_bytes, {
            'provider': 'openai',
            'request_id': response.headers.get('x-request-id') or response.headers.get('X-Request-Id'),
            'content_type': response.headers.get('Content-Type') or response.headers.get('content-type'),
        }

    def _decode_volcengine_events(self, response: requests.Response) -> Iterable[Dict[str, Any]]:
        response.encoding = 'utf-8'
        text = response.text
        payload_parts = self._sse_data_parts(text)
        if not payload_parts:
            payload_parts = [text]

        decoder = json.JSONDecoder()
        for payload in payload_parts:
            index = 0
            while index < len(payload):
                while index < len(payload) and payload[index].isspace():
                    index += 1
                if index >= len(payload):
                    break
                try:
                    event, index = decoder.raw_decode(payload, index)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f'Failed to decode Volcengine TTS response payload: {payload!r}') from e
                if isinstance(event, dict):
                    yield event

    @staticmethod
    def _sse_data_parts(text: str) -> List[str]:
        parts: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.startswith('data:'):
                continue
            data = line.removeprefix('data:').strip()
            if data and data != '[DONE]':
                parts.append(data)
        return parts
