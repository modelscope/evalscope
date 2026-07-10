from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import AbstractAsyncContextManager
from typing import IO, Optional

from evalscope.perf.config.models import TargetConfig
from evalscope.perf.domain.errors import PerfRunError


class ManagedTarget(AbstractAsyncContextManager):
    """Own the lifecycle of an optional local inference target."""

    def __init__(self, target: TargetConfig, log_path: str) -> None:
        self.target = target
        self.log_path = log_path
        self._process: Optional[subprocess.Popen] = None
        self._log: Optional[IO] = None
        self._server = None
        self._server_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> 'ManagedTarget':
        if self.target.kind == 'remote':
            return self
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.target.kind == 'local_vllm':
            await self._start_vllm()
        else:
            await self._start_transformers()
        return self

    async def _start_vllm(self) -> None:
        try:
            import torch
        except ImportError as e:
            raise PerfRunError('target', 'startup', 'torch is required for local_vllm') from e
        self._log = open(self.log_path, 'a', encoding='utf-8')
        env = dict(os.environ)
        env.update({
            'VLLM_USE_MODELSCOPE': 'True',
            'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1',
            'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        })
        command = [
            sys.executable,
            '-m',
            'vllm.entrypoints.openai.api_server',
            '--model',
            self.target.model,
            '--served-model-name',
            self.target.model,
            '--tensor-parallel-size',
            str(max(1, torch.cuda.device_count())),
            '--host',
            '127.0.0.1',
            '--port',
            str(self.target.port),
            '--trust-remote-code',
            '--disable-log-requests',
            '--disable-log-stats',
        ]
        try:
            self._process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT, env=env)
        except Exception as e:
            self._log.close()
            self._log = None
            raise PerfRunError('target', 'startup', f'failed to start local vLLM: {e}') from e

    async def _start_transformers(self) -> None:
        try:
            import uvicorn
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            from sse_starlette.sse import EventSourceResponse

            from evalscope.utils.chat_service import (
                ChatCompletionRequest,
                ChatService,
                ModelList,
                TextCompletionRequest,
            )
        except ImportError as e:
            raise PerfRunError('target', 'startup', f'local target dependency is missing: {e}') from e

        service = await asyncio.to_thread(
            ChatService,
            model_path=self.target.model,
            attn_implementation=self.target.attn_implementation,
        )
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )

        @app.get('/v1/models', response_model=ModelList)
        async def list_models():
            return await service.list_models()

        @app.post('/v1/completions')
        async def completions(request: TextCompletionRequest):
            return await service._text_completion(request)

        @app.post('/v1/chat/completions')
        async def chat(request: ChatCompletionRequest):
            if request.stream:
                return EventSourceResponse(service._stream_chat(request))
            return await service._chat(request)

        config = uvicorn.Config(app, host='127.0.0.1', port=self.target.port, log_level='info')
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())

    def ensure_running(self) -> None:
        if self._process is not None and self._process.poll() is not None:
            raise PerfRunError('target', 'startup', f'local vLLM exited with code {self._process.returncode}')
        if self._server_task is not None and self._server_task.done():
            error = self._server_task.exception()
            raise PerfRunError('target', 'startup', str(error or 'local server stopped'))

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._server is not None and self._server_task is not None:
            self._server.should_exit = True
            try:
                await asyncio.wait_for(self._server_task, timeout=10)
            except asyncio.TimeoutError:
                self._server_task.cancel()
                await asyncio.gather(self._server_task, return_exceptions=True)
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self._process.wait), timeout=10)
            except asyncio.TimeoutError:
                self._process.kill()
                await asyncio.to_thread(self._process.wait)
        if self._log is not None:
            self._log.close()
