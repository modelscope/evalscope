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

    VOLCENGINE_TTS_URL = 'https://openspeech.bytedance.com/api/v3/tts/unidirectional'
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

        self.provider = model_args.pop('provider', 'volcengine')
        if self.provider != 'volcengine':
            raise ValueError(f'Unsupported text2speech provider: {self.provider}')

        self.base_url = (base_url or self.VOLCENGINE_TTS_URL).rstrip('/')
        self.api_key = self._resolve_api_key(api_key)
        self.resource_id = model_args.pop('resource_id', model_name)
        self.speaker = model_args.pop('speaker', None)
        self.user_id = model_args.pop('user_id', 'evalscope')
        self.timeout = model_args.pop('timeout', 60)
        self.audio_format = model_args.pop('format', 'mp3')
        self.audio_params = model_args.pop('audio_params', {})
        self.req_params = model_args.pop('req_params', {})
        self.app_id = model_args.pop('app_id', None)
        self.access_key = model_args.pop('access_key', None)
        self.session = requests.Session()

        if not self.api_key and not (self.app_id and self.access_key):
            raise ValueError('api_key is required for Volcengine text2speech. Set --api-key or VOLCENGINE_TTS_API_KEY.')
        if not self.speaker and 'speaker' not in self.req_params:
            raise ValueError('model_args.speaker is required for Volcengine text2speech.')
        if self.audio_format not in {'mp3', 'wav'}:
            raise ValueError('text2speech output format must be "mp3" or "wav".')

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        start_time = time.monotonic()
        payload = self._volcengine_payload(input)
        headers = self._volcengine_headers()
        response = self.session.post(
            self.base_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=config.timeout or self.timeout,
        )
        response.raise_for_status()
        audio_bytes, metadata = self._collect_volcengine_audio(response)

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
            'log_id': response.headers.get('X-Tt-Logid') or response.headers.get('X-Tt-Logid'.lower()),
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

    def _decode_volcengine_events(self, response: requests.Response) -> Iterable[Dict[str, Any]]:
        text = self._response_text(response)
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
                event, index = decoder.raw_decode(payload, index)
                if isinstance(event, dict):
                    yield event

    @staticmethod
    def _response_text(response: requests.Response) -> str:
        chunks: List[str] = []
        for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
            if not chunk:
                continue
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8')
            chunks.append(chunk)
        if chunks:
            return ''.join(chunks)
        return response.text

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

    @staticmethod
    def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
        if api_key and api_key != 'EMPTY':
            return api_key
        return os.getenv('VOLCENGINE_TTS_API_KEY') or os.getenv('VOLC_TTS_API_KEY')
