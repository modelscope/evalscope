"""Example EvalScope plugin.

Importing this module registers, through the existing EvalScope decorators:

- model API ``example_arith_api`` (``evalscope eval --eval-type``)
- benchmark ``example_arith`` (``evalscope eval --datasets``)
- metric ``example_numeric_match`` (used by the ``example_arith`` benchmark)
- perf API protocol ``example_arith`` (``evalscope perf --api``)
- perf dataset ``example_arith_prompts`` (``evalscope perf --dataset``)

All of them talk to the toy HTTP endpoint in :mod:`evalscope_example_plugin.server`, which speaks
a small non-OpenAI "arith" protocol.
"""

import json
import os
import re
import sys
import time
import traceback
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import aiohttp

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessage
from evalscope.api.metric import Metric
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.model.model_output import ModelUsage
from evalscope.api.registry import register_benchmark, register_metric, register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.constants import Tags
from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_api, register_dataset
from evalscope.perf.utils.benchmark_util import BenchmarkData

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'test.jsonl')


# --- Model API: one request per sample, no retries; raw response and request id kept in metadata.


@register_model_api('example_arith_api')
class ExampleArithAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)
        if not base_url:
            raise ValueError('example_arith_api needs --api-url, e.g. http://127.0.0.1:8901/v1')
        self.endpoint = base_url.rstrip('/') + '/arith'
        self.timeout = float(model_args.get('timeout', 30))

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        body = {'model': self.model_name, 'prompt': '\n'.join(message.text for message in input)}
        if config.max_tokens is not None:
            body['max_tokens'] = config.max_tokens
        headers = {'Content-Type': 'application/json'}
        if self.api_key and self.api_key != 'EMPTY':
            headers['Authorization'] = f'Bearer {self.api_key}'
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST'
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode('utf-8')
            request_id = response.headers.get('X-Request-Id')
        payload = json.loads(raw)
        usage = payload.get('usage', {})
        output = ModelOutput.from_content(model=self.model_name, content=payload['text'])
        output.usage = ModelUsage(
            input_tokens=usage.get('input_tokens', 0),
            output_tokens=usage.get('output_tokens', 0),
            total_tokens=usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
        )
        output.metadata = {'raw_response': raw, 'request_id': request_id}
        return output


# --- Metric: compare the last number in the prediction with the reference number.

_NUMBER = re.compile(r'-?\d+(?:\.\d+)?')


@register_metric(name='example_numeric_match')
class ExampleNumericMatch(Metric):
    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        scores = []
        for prediction, reference in zip(predictions, references):
            numbers = _NUMBER.findall(prediction or '')
            scores.append(1.0 if numbers and float(numbers[-1]) == float(reference) else 0.0)
        return scores


# --- Benchmark: a small arithmetic set shipped with the plugin.

DESCRIPTION = """
## Overview
A 20-question arithmetic set that demonstrates a benchmark shipped by an EvalScope plugin.

## Task Description
- **Task Type**: Arithmetic question answering
- **Input**: A question such as "What is 42 - 20?"
- **Output**: A sentence containing the numeric answer
- **Domain**: Integer arithmetic

## Key Features
- Bundled with the plugin package; nothing is downloaded.

## Evaluation Notes
- Scored with the plugin's `example_numeric_match` metric, which compares the last number in the answer.
"""


@register_benchmark(
    BenchmarkMeta(
        name='example_arith',
        evaluation_version='v1.0',
        pretty_name='Example Arithmetic',
        dataset_id=DATA_DIR,
        tags=[Tags.MATH],
        description=DESCRIPTION,
        subset_list=['default'],
        metric_list=['example_numeric_match'],
        eval_split='test',
        few_shot_num=0,
        prompt_template='{question}',
    )
)
class ExampleArithAdapter(DefaultDataAdapter):
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(input=record['question'], target=record['answer'])


# --- Perf API protocol and perf dataset for the same endpoint.


@register_api('example_arith')
class ExampleArithPerfPlugin(ApiPluginBase):
    """Sends ``{"model", "prompt", "max_tokens"}`` and reads ``{"text", "usage"}`` back."""

    def build_request(self, messages: Union[List[Dict], str], param: Optional[Arguments] = None) -> Dict:
        param = param or self.param
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = '\n'.join(str(message.get('content', '')) for message in messages)
        body: Dict[str, Any] = {'model': param.model, 'prompt': prompt}
        if isinstance(param.max_tokens, int):
            body['max_tokens'] = param.max_tokens
        return body

    def parse_responses(self, responses: List[Dict], request: str = None, **kwargs: Any) -> Tuple[int, int]:
        usage = (responses[-1] if responses else {}).get('usage', {})
        return usage.get('input_tokens', 0), usage.get('output_tokens', 0)

    async def process_request(
        self, client_session: aiohttp.ClientSession, url: str, headers: Dict, body: Dict
    ) -> BenchmarkData:
        output = BenchmarkData()
        output.request = json.dumps(body, ensure_ascii=False)
        start = time.perf_counter()
        output.start_time = start
        try:
            async with client_session.post(url, json=body, headers=headers) as response:
                if response.status != 200:
                    output.status_code = response.status
                    output.error = await response.text()
                else:
                    payload = await response.json()
                    output.response_messages.append(payload)
                    output.generated_text = payload.get('text', '')
                    output.prompt_tokens, output.completion_tokens = self.parse_responses([payload])
                    output.success = True
        except Exception:
            output.error = ''.join(traceback.format_exception(*sys.exc_info()))
        output.completed_time = time.perf_counter()
        output.query_latency = output.completed_time - start
        # The protocol is not streamed, so the first output arrives with the whole response.
        output.first_chunk_latency = output.query_latency
        return output


@register_dataset('example_arith_prompts')
class ExampleArithPrompts(DatasetPluginBase):
    """Questions from the plugin's bundled arithmetic set, one request each."""

    requires_model = True

    def build_messages(self) -> Iterator[List[Dict]]:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                messages = self.prepare_messages(json.loads(line)['question'])
                if messages is not None:
                    yield messages
