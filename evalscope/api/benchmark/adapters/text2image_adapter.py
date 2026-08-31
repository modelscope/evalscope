import base64
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.messages.content import ContentImage
from evalscope.api.metric import AggScore, MetricIdentity, MetricKind, MetricSelector, SampleScore, Score
from evalscope.api.model import ChatCompletionChoice, Model, ModelOutput
from evalscope.api.registry import get_metric
from evalscope.constants import EvalType, FileConstants
from evalscope.metrics.semantics import get_semantics_resolver
from evalscope.metrics.semantics.identity import canonicalize_producer_identity
from evalscope.utils import get_logger
from evalscope.utils.function_utils import thread_safe

from .default_data_adapter import DefaultDataAdapter

logger = get_logger()

_METRIC_IDENTITIES_KEY = 'metric_identities'
_DIMENSION_VALUE = re.compile(r'[^a-z0-9]+')

T2I_REPORT_METRIC_NAMES = {
    'BLIPv2Score': 'blipv2_score',
    'CLIPScore': 'clipscore',
    'FGA_BLIP2Score': 'fga_blip2_score',
    'HPSv2.1Score': 'hps_v2_1_score',
    'HPSv2Score': 'hpsv2_score',
    'ImageRewardScore': 'image_reward_score',
    'MPS': 'mps',
    'PickScore': 'pick_score',
    'VQAScore': 'vqa_model_score',
}
"""T2I scorer registry name -> canonical report metric name."""

IMAGE_PAIR_REFERENCE_KEYS = (
    'reference_image_path',
    'target_image_path',
    'ref_image_path',
    'gt_image_path',
    'ground_truth_image_path',
    'reference_image',
    'target_image',
    'ref_image',
    'gt_image',
    'ground_truth_image',
)


class Text2ImageAdapter(DefaultDataAdapter):
    """Text to Image Adapter for benchmarks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.primary_metric is None:
            self._benchmark_meta.primary_metric = self._default_primary_metric()

    @staticmethod
    def canonical_metric_name(metric_name: str) -> str:
        """Return the report metric name produced by a T2I scorer."""
        explicit = T2I_REPORT_METRIC_NAMES.get(metric_name)
        if explicit is not None:
            return explicit
        return canonicalize_producer_identity(metric_name, 'identity').name

    @staticmethod
    def _dimension_value(value: Any) -> str:
        """Normalize a dynamic category/component into a stable dimension value."""
        return _DIMENSION_VALUE.sub('_', str(value).strip().lower()).strip('_')

    def _default_primary_metric(self) -> MetricSelector | None:
        """Select the first configured T2I metric with declared semantics."""
        resolver = get_semantics_resolver()
        for entry in self.metric_list:
            scorer_name = entry if isinstance(entry, str) else next(iter(entry), '')
            metric_name = self.canonical_metric_name(scorer_name)
            identity = MetricIdentity(name=metric_name, aggregation='mean', dimensions={'scope': 'overall'})
            if resolver.resolve(self.name, identity).semantics.kind is not MetricKind.DIAGNOSTIC:
                return MetricSelector(name=metric_name, aggregation='mean', dimensions={'scope': 'overall'})
        return None

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a record dictionary to a Sample object."""
        metadata = dict(record)
        metadata.update(
            {
                'prompt': record['prompt'],
                'category': record.get('category', ''),
                'tags': record.get('tags', []),
                FileConstants.ID: record.get(FileConstants.ID, ''),
                FileConstants.IMAGE_PATH: record.get(FileConstants.IMAGE_PATH, ''),
            }
        )
        return Sample(
            input=[ChatMessageUser(content=record['prompt'])],
            target=self._record_reference_image(record),
            metadata=metadata,
        )

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        """
        Hook method called during the actual inference process.

        This method executes the model inference and can be overridden
        to implement custom inference logic or model interaction patterns.

        Args:
            model (Model): The model to use for inference
            sample (Sample): The sample to process

        Returns:
            ModelOutput: The raw output from the model
        """
        if self.eval_type == EvalType.MOCK_LLM:
            return ModelOutput(
                model=model.name,
                choices=[ChatCompletionChoice.from_content('')],
            )
        else:
            # Execute model inference with the processed input and any tools
            model_output = model.generate(input=sample.input, tools=sample.tools)
            return model_output

    def _on_inference_end(
        self, model: Model, sample: Sample, model_output: ModelOutput, output_dir: str, **kwargs
    ) -> TaskState:
        """
        Hook method called after inference completes. Save generated images to output_dir.

        Args:
            model (Model): The model that performed inference
            sample (Sample): The processed sample
            model_output (ModelOutput): The raw model output
            output_dir (str): The directory where the model output was saved

        Returns:
            TaskState: Complete state object for the inference task
        """
        if self.eval_type == EvalType.MOCK_LLM:
            return TaskState(
                model=model.name,
                sample=sample,
                messages=[model_output.message],
                output=model_output,
                completed=True,
            )
        else:
            image_id = f'{sample.metadata.get(FileConstants.ID) or sample.id}_{sample.group_id}'
            output_path = os.path.join(output_dir, 'images', f'{image_id}.png')
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            # get base64 image from model_output
            content = model_output.message.content[0]

            assert isinstance(content, ContentImage), 'Expected ContentImage in model output'

            image_base64 = content.image
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(image_base64))

            sample.metadata[FileConstants.IMAGE_PATH] = output_path
            return TaskState(
                model=model.name,
                sample=sample,
                messages=[model_output.message],
                output=model_output,
                completed=True,
            )

    # NOTE: thread safe is needed, since we can't batch inference here.
    @thread_safe
    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        # Get prediction and prompt from task state
        meta = task_state.metadata or {}
        image_path = meta.get(FileConstants.IMAGE_PATH, original_prediction)
        if isinstance(task_state.input, list) and task_state.input:
            prompt = task_state.input[0].content
        else:
            prompt = task_state.input

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=image_path,
            prediction=image_path,
        )

        # Calculate scores for each configured metric
        for metric in self.metric_list:
            metric_name = ''
            try:
                if isinstance(metric, str):
                    metric_name = metric
                elif isinstance(metric, dict):
                    metric_name = list(metric.keys())[0]
                else:
                    continue
                metric_args = self.get_metric_args(metric_name)
                metric_cls = get_metric(metric_name)
                metric_func = metric_cls(**metric_args)
                if self._is_image_pair_metric(metric_func):
                    reference_image = self._resolve_reference_image(reference, meta, metric_name)
                    metric_score = metric_func(image_path, reference_image)
                else:
                    metric_score = metric_func(image_path, prompt)[0]

                self._record_metric_result(score, metric_name, metric_score)
            except Exception as e:
                logger.error(f'Error calculating metric {metric}: {e}')
                metric_name = self.canonical_metric_name(metric_name)
                self._record_metric_result(score, metric_name, 0)
                score.metadata[metric_name] = f'error: {str(e)}'

        return score

    def _record_metric_result(self, score: Score, scorer_name: str, metric_score: Any) -> None:
        """Store scorer output together with its canonical report identity."""
        metric_name = self.canonical_metric_name(scorer_name)
        identities = score.metadata.setdefault(_METRIC_IDENTITIES_KEY, {})

        if not isinstance(metric_score, dict):
            score.value[metric_name] = self._score_to_float(metric_score)
            identities[metric_name] = {'name': metric_name, 'dimensions': {}}
            return

        for raw_component, value in metric_score.items():
            component = self._dimension_value(raw_component)
            dimensions = {} if component in {'overall', 'overall_score'} else {'component': component}
            storage_key = metric_name if not dimensions else f'{metric_name}:{component}'
            score.value[storage_key] = self._score_to_float(value)
            identities[storage_key] = {'name': metric_name, 'dimensions': dimensions}

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate T2I scores into canonical overall and category identities."""
        grouped: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], List[Tuple[float, Any]]] = defaultdict(list)

        for sample_score in sample_scores:
            metadata = sample_score.score.metadata or {}
            identities = metadata.get(_METRIC_IDENTITIES_KEY, {})
            category = self._dimension_value((sample_score.sample_metadata or {}).get('category', ''))

            for storage_key, value in sample_score.score.value.items():
                descriptor = identities.get(storage_key)
                if descriptor is None:
                    identity = canonicalize_producer_identity(storage_key, 'mean')
                    metric_name = identity.name
                    dimensions = dict(identity.dimensions)
                else:
                    metric_name = descriptor['name']
                    dimensions = dict(descriptor.get('dimensions') or {})

                overall_dimensions = dimensions if 'component' in dimensions else {**dimensions, 'scope': 'overall'}
                grouped[(metric_name, tuple(sorted(overall_dimensions.items())))].append(
                    (float(value), sample_score.sample_id)
                )
                if category:
                    category_dimensions = {**dimensions, 'category': category}
                    grouped[(metric_name, tuple(sorted(category_dimensions.items())))].append(
                        (float(value), sample_score.sample_id)
                    )

        return [
            AggScore(
                metric_name=metric_name,
                aggregation='mean',
                dimensions=dict(dimensions),
                score=sum(value for value, _ in values) / len(values),
                num=len(values),
                ids=[sample_id for _, sample_id in values],
            )
            for (metric_name, dimensions), values in grouped.items()
        ]

    @staticmethod
    def _record_reference_image(record: Dict[str, Any]) -> str:
        for key in IMAGE_PAIR_REFERENCE_KEYS:
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
        target = record.get('target', '')
        return target if isinstance(target, str) else ''

    @staticmethod
    def _is_image_pair_metric(metric_func: Any) -> bool:
        return bool(getattr(metric_func, 'image_pair_metric', False))

    @staticmethod
    def _resolve_reference_image(reference: Any, metadata: Dict[str, Any], metric_name: str) -> Any:
        for key in IMAGE_PAIR_REFERENCE_KEYS:
            value = metadata.get(key)
            if not Text2ImageAdapter._is_empty_image_value(value):
                return value
        if not Text2ImageAdapter._is_empty_image_value(reference):
            return reference
        raise ValueError(
            f'Metric {metric_name} requires a reference image. Provide one of: {", ".join(IMAGE_PAIR_REFERENCE_KEYS)}.'
        )

    @staticmethod
    def _is_empty_image_value(value: Any) -> bool:
        return value is None or (isinstance(value, str) and value.strip() == '')

    @staticmethod
    def _score_to_float(value: Any) -> float:
        if hasattr(value, 'cpu') and hasattr(value, 'item'):
            return float(value.cpu().item())
        if hasattr(value, 'item'):
            return float(value.item())
        return float(value)
