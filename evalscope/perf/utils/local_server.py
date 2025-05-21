import os
import subprocess
import torch
import uvicorn
from contextlib import asynccontextmanager
from dataclasses import dataclass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from evalscope.perf.arguments import Arguments
from evalscope.utils.chat_service import ChatCompletionRequest, ChatService, ModelList, TextCompletionRequest
from evalscope.utils.logger import get_logger

logger = get_logger()


@dataclass
class ServerSentEvent(object):

    def __init__(self, data='', event=None, id=None, retry=None):
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry

    @classmethod
    def decode(cls, line):
        """Decode line to ServerSentEvent


        Args:
            line (str): The line.

        Return:
            ServerSentEvent (obj:`ServerSentEvent`): The ServerSentEvent object.

        """
        if not line:
            return None
        sse_msg = cls()
        # format data:xxx
        field_type, _, field_value = line.partition(':')
        if field_value.startswith(' '):  # compatible with openai api
            field_value = field_value[1:]
        if field_type == 'event':
            sse_msg.event = field_value
        elif field_type == 'data':
            field_value = field_value.rstrip()
            sse_msg.data = field_value
        elif field_type == 'id':
            sse_msg.id = field_value
        elif field_type == 'retry':
            sse_msg.retry = field_value
        else:
            pass

        return sse_msg


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_app(model, attn_implementation=None) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    chat_service = ChatService(model_path=model, attn_implementation=attn_implementation)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.get('/v1/models', response_model=ModelList)
    async def list_models():
        return await chat_service.list_models()

    @app.post('/v1/completions')
    async def create_text_completion(request: TextCompletionRequest):
        return await chat_service._text_completion(request)

    @app.post('/v1/chat/completions')
    async def create_chat_completion(request: ChatCompletionRequest):
        if request.stream:
            return EventSourceResponse(chat_service._stream_chat(request))
        else:
            return await chat_service._chat(request)

    return app


def start_app(args: Arguments):
    logger.info('Starting local server, please wait...')
    if args.api == 'local':
        app = create_app(args.model, args.attn_implementation)
        uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)

    elif args.api == 'local_vllm':
        os.environ['VLLM_USE_MODELSCOPE'] = 'True'
        os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        # yapf: disable
        proc = subprocess.Popen([
            'python', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', args.model,
            '--served-model-name', args.model,
            '--tensor-parallel-size', str(torch.cuda.device_count()),
            '--max-model-len', '32768',
            '--gpu-memory-utilization', '0.9',
            '--host', '0.0.0.0',
            '--port', str(args.port),
            '--trust-remote-code',
            '--disable-log-requests',
            '--disable-log-stats',
        ])
        # yapf: enable
        import atexit

        def on_exit():
            if proc.poll() is None:
                logger.info('Terminating the child process...')
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning('Child process did not terminate within the timeout, killing it forcefully...')
                    proc.kill()
                    proc.wait()
                logger.info('Child process terminated.')
            else:
                logger.info('Child process has already terminated.')

        atexit.register(on_exit)
    else:
        raise ValueError(f'Unknown API type: {args.api}')


if __name__ == '__main__':
    from collections import namedtuple

    args = namedtuple('Args', ['model', 'attn_implementation', 'api'])

    start_app(args(model='Qwen/Qwen2.5-0.5B-Instruct', attn_implementation=None, api='local_vllm'))
