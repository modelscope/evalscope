import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.argument_utils import get_secret_value
from .client import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WS_PROXY_PORT,
    ToolathlonServiceClient,
    ToolathlonServiceConfig,
)

DEFAULT_TASK_LIST = [
    'ab-testing',
    'academic-pdf-report',
    'academic-warning',
    'add-bibtex',
    'apply-phd-email',
    'arrange-workspace',
    'canvas-arrange-exam',
    'canvas-art-manager',
    'canvas-art-quiz',
    'canvas-do-quiz',
    'canvas-homework-grader-python',
    'canvas-list-test',
    'canvas-new-students-notification',
    'canvas-submit-late-work',
    'cooking-guidance',
    'course-assistant',
    'course-schedule',
    'courses-ta-hws',
    'cvpr-research',
    'dataset-license-issue',
    'detect-revised-terms',
    'dietary-health',
    'email-paper-homepage',
    'excel-data-transformation',
    'excel-market-research',
    'experiments-recordings',
    'fillout-online-forms',
    'filter-low-selling-products',
    'find-alita-paper',
    'flagged-transactions',
    'game-statistics',
    'gdp-cr5-analysis',
    'git-bug-hunt',
    'git-milestone',
    'git-repo',
    'hk-top-conf',
    'huggingface-upload',
    'identify-all-songs',
    'imagenet',
    'inter-final-performance-analysis',
    'interview-report',
    'inventory-sync',
    'investment-decision-analysis',
    'invoice-org',
    'ipad-edu-price',
    'k8s-deployment-cleanup',
    'k8s-mysql',
    'k8s-pr-preview-testing',
    'k8s-redis-helm-upgrade',
    'k8s-safety-audit',
    'landing-task-reminder',
    'language-school',
    'latex-prompt-box',
    'live-transactions',
    'llm-training-dataset',
    'logical-datasets-collection',
    'machine-operating',
    'meeting-assign',
    'merge-hf-datasets',
    'mrbeast-analysis',
    'music-analysis',
    'nhl-b2b-analysis',
    'notion-find-job',
    'notion-hr',
    'notion-movies',
    'notion-personal-website',
    'nvidia-market',
    'nvidia-stock-analysis',
    'oil-price',
    'paper-checker',
    'payable-invoice-checker',
    'personal-website-construct',
    'ppt-analysis',
    'price-comparison',
    'privacy-desensitization',
    'profile-update-online',
    'quantitative-financial-analysis',
    'reimbursement-form-filler',
    'sales-accounting',
    'search-ca-school',
    'set-conf-cr-ddl',
    'shopping-helper',
    'sla-timeout-monitor',
    'stock-build-position',
    'student-interview',
    'subway-planning',
    'sync-todo-to-readme',
    'task-tracker',
    'train-ticket-plan',
    'travel-exchange',
    'travel-expense-reimbursement',
    'trip-adviser',
    'trip-itinerary-generator',
    'university-course-selection',
    'update-material-inventory',
    'upenn-campus-route',
    'verl-dataset',
    'vlm-history-completer',
    'wandb-best-score',
    'wandb-shortest-length',
    'woocommerce-customer-survey',
    'woocommerce-new-product',
    'woocommerce-new-welcome',
    'woocommerce-product-recall',
    'woocommerce-stock-alert',
    'woocommerce-update-cover',
    'yahoo-analysis',
    'youtube-repo',
]
MODEL_PARAM_EXCLUDES = {
    'model',
    'messages',
    'tools',
    'tool_choice',
    'stream',
    'batch_size',
    'retries',
    'retry_interval',
    'anthropic_cache_strategy',
}

DESCRIPTION = """
## Overview

Toolathlon is an agent benchmark for realistic, long-horizon tool use across many MCP-backed software
environments. This EvalScope benchmark is a wrapper around the official Toolathlon remote evaluation service,
not a local reimplementation of the MCP environments or official evaluator.

## Evaluation Mode

- Benchmark id: `toolathlon`
- Supported mode: official service private mode
- EvalScope controls model endpoint, task selection, job parameters, polling, result download, and reporting
- The official Toolathlon service controls MCP environments, task containers, agent loop execution, and scoring
- No Toolathlon, MCP application accounts, or Toolathlon Python package installation are required when using the
  official public evaluation service
- A local or intranet OpenAI-compatible endpoint is required for private mode
- EvalScope represents one remote Toolathlon job as one local sample; the generated data statistics count wrapper
  jobs, while `task_list` and `limit` control the Toolathlon tasks submitted inside that job
- The bundled Toolathlon-Verified task list was inspected from official repository commit
  `b7bbac3f9a1f381b095c878debe1a47dd164ad85`

## Usage Guide

See the Toolathlon usage guide for public-service limits, private-mode data flow, self-hosted service setup, and
EvalScope configuration examples:

- https://evalscope.readthedocs.io/en/latest/third_party/toolathlon.html

Official sources:

- https://github.com/hkust-nlp/Toolathlon
- https://github.com/hkust-nlp/Toolathlon/blob/main/EVAL_SERVICE_README.md
"""

EXTRA_PARAMS = {
    'mode': {
        'type': 'str',
        'description': 'Toolathlon service mode. EvalScope supports private mode for this wrapper.',
        'value': 'private',
        'choices': ['private'],
    },
    'server_host': {
        'type': 'str',
        'description': 'Official Toolathlon evaluation service host.',
        'value': DEFAULT_SERVER_HOST,
    },
    'server_port': {
        'type': 'int',
        'description': 'Official Toolathlon HTTP service port.',
        'value': DEFAULT_SERVER_PORT,
    },
    'ws_proxy_port': {
        'type': 'int',
        'description': 'Official Toolathlon WebSocket proxy port for private mode.',
        'value': DEFAULT_WS_PROXY_PORT,
    },
    'workers': {
        'type': 'int',
        'description': 'Number of parallel Toolathlon workers requested from the official service.',
        'value': 10,
    },
    'provider': {
        'type': 'str',
        'description': 'Toolathlon model provider type.',
        'value': 'unified',
        'choices': ['unified', 'openai_stateful_responses'],
    },
    'task_list': {
        'type': 'list',
        'description': 'Optional Toolathlon task names to evaluate. Empty uses the bundled Toolathlon-Verified list.',
        'value': [],
    },
    'task_list_file': {
        'type': 'str',
        'description': 'Optional file containing one Toolathlon task name per line.',
        'value': '',
    },
    'model_params': {
        'type': 'dict',
        'description': 'Extra model parameters forwarded to Toolathlon, merged after TaskConfig.generation_config.',
        'value': {},
    },
    'job_id': {
        'type': 'str',
        'description': 'Optional Toolathlon job id. Reuse to resume an incomplete official-service job.',
        'value': '',
    },
    'force_redownload': {
        'type': 'bool',
        'description': 'Force redownload of Toolathlon result archives.',
        'value': False,
    },
    'override_output_dir': {
        'type': 'bool',
        'description': 'Clear the Toolathlon output directory when it already contains files.',
        'value': False,
    },
    'skip_container_restart': {
        'type': 'bool',
        'description': 'Skip Toolathlon container restart. Use only for small debug task subsets.',
        'value': False,
    },
    'trust_env_in_httpx': {
        'type': 'bool',
        'description': 'Allow httpx to use proxy environment variables.',
        'value': False,
    },
    'timeout_seconds': {
        'type': 'int',
        'description': 'Maximum time to wait for a Toolathlon official-service job.',
        'value': DEFAULT_TIMEOUT_SECONDS,
    },
    'poll_interval': {
        'type': 'int',
        'description': 'Polling interval in seconds for Toolathlon job status.',
        'value': DEFAULT_POLL_INTERVAL_SECONDS,
    },
}


@register_benchmark(
    BenchmarkMeta(
        name='toolathlon',
        pretty_name='Toolathlon Official Service Wrapper',
        dataset_id='https://github.com/hkust-nlp/Toolathlon',
        tags=[Tags.AGENT, Tags.FUNCTION_CALLING, Tags.MULTI_TURN],
        description=DESCRIPTION,
        metric_list=['acc'],
        eval_split='test',
        subset_list=['default'],
        prompt_template='{question}',
        extra_params=EXTRA_PARAMS,
    )
)
class ToolathlonAdapter(AgentAdapter):
    """EvalScope wrapper around the official Toolathlon private-mode service."""

    client_cls = ToolathlonServiceClient

    def load(self) -> tuple[DatasetDict, None]:
        task_list = self._resolve_task_list()
        sample = Sample(
            input='Run Toolathlon official remote evaluation service.',
            target='',
            id=0,
            metadata={
                'task_list': task_list,
                'mode': self.extra_params.get('mode', 'private'),
            },
        )
        dataset = MemoryDataset([sample], name='toolathlon', location=self.dataset_id)
        return DatasetDict({'default': dataset}), None

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        config = self._build_service_config(model, sample)
        result = self.client_cls(config).run_private()
        sample.metadata['toolathlon_result'] = result

        content = json.dumps(
            {
                'job_id': result.get('job_id'),
                'output_dir': result.get('output_dir'),
                'acc': result.get('acc', 0.0),
            },
            ensure_ascii=False,
        )
        output = ModelOutput.from_content(model=model.name, content=content)
        output.metadata = result
        return InferenceResult(output=output)

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: Any,
    ) -> Score:
        result = task_state.metadata.get('toolathlon_result', {})
        acc = float(result.get('acc', 0.0))
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'acc': acc},
            main_score_name='acc',
            metadata=result,
        )

    def _build_service_config(self, model: Model, sample: Sample) -> ToolathlonServiceConfig:
        mode = self.extra_params.get('mode', 'private')
        if mode != 'private':
            raise ValueError('The EvalScope Toolathlon wrapper supports only private mode.')

        base_url = self._resolve_base_url(model)
        if not base_url:
            raise ValueError('Toolathlon private mode requires TaskConfig.api_url or model.api.base_url.')

        output_dir = Path(self.output_dir) / 'toolathlon'
        return ToolathlonServiceConfig(
            server_host=str(self.extra_params.get('server_host', DEFAULT_SERVER_HOST)),
            server_port=int(self.extra_params.get('server_port', DEFAULT_SERVER_PORT)),
            ws_proxy_port=int(self.extra_params.get('ws_proxy_port', DEFAULT_WS_PROXY_PORT)),
            base_url=base_url.rstrip('/'),
            model_name=model.name,
            api_key=self._resolve_api_key(model),
            workers=int(self.extra_params.get('workers', 10)),
            provider=str(self.extra_params.get('provider', 'unified')),
            task_list=sample.metadata.get('task_list'),
            model_params=self._resolve_model_params(model),
            job_id=self._empty_to_none(self.extra_params.get('job_id')),
            force_redownload=bool(self.extra_params.get('force_redownload', False)),
            override_output_dir=bool(self.extra_params.get('override_output_dir', False)),
            skip_container_restart=bool(self.extra_params.get('skip_container_restart', False)),
            trust_env_in_httpx=bool(self.extra_params.get('trust_env_in_httpx', False)),
            timeout_seconds=int(self.extra_params.get('timeout_seconds', DEFAULT_TIMEOUT_SECONDS)),
            poll_interval=int(self.extra_params.get('poll_interval', DEFAULT_POLL_INTERVAL_SECONDS)),
            output_dir=output_dir,
        )

    def _resolve_base_url(self, model: Model) -> str:
        if self._task_config is not None and self._task_config.api_url:
            return self._task_config.api_url
        return getattr(model.api, 'base_url', '') or ''

    def _resolve_api_key(self, model: Model) -> Optional[str]:
        if self._task_config is not None and self._task_config.api_key:
            return get_secret_value(self._task_config.api_key)
        return getattr(model.api, 'api_key', None)

    def _resolve_model_params(self, model: Model) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        generation_config = getattr(model, 'config', None)
        if generation_config is not None:
            if hasattr(generation_config, 'model_dump'):
                params.update(generation_config.model_dump(exclude_none=True))
            elif isinstance(generation_config, dict):
                params.update(generation_config)
        params = {key: value for key, value in params.items() if key not in MODEL_PARAM_EXCLUDES}
        params.update(self.extra_params.get('model_params') or {})
        return params

    def _resolve_task_list(self) -> List[str]:
        task_list = self.extra_params.get('task_list') or []
        task_list_file = self.extra_params.get('task_list_file') or ''
        if task_list_file:
            task_path = Path(task_list_file)
            task_list = [line.strip() for line in task_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        task_list = [str(task).strip() for task in task_list if str(task).strip()]
        if not task_list:
            task_list = list(DEFAULT_TASK_LIST)
        if isinstance(self.limit, int) and self.limit > 0:
            task_list = task_list[:self.limit]
        return task_list

    def _empty_to_none(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value or None
