import base64
import os
from typing import Any, Dict

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.messages.content import ContentImage
from evalscope.api.metric import Score
from evalscope.api.model import ChatCompletionChoice, Model, ModelOutput
from evalscope.api.registry import get_metric
from evalscope.constants import EvalType, FileConstants
from evalscope.utils import get_logger
from evalscope.utils.function_utils import thread_safe
from .default_data_adapter import DefaultDataAdapter

logger = get_logger()

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

        self.add_aggregation_name = False  # Do not add aggregation name in the report by default

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a record dictionary to a Sample object."""
        metadata = dict(record)
        metadata.update({
            'prompt': record['prompt'],
            'category': record.get('category', ''),
            'tags': record.get('tags', []),
            FileConstants.ID: record.get(FileConstants.ID, ''),
            FileConstants.IMAGE_PATH: record.get(FileConstants.IMAGE_PATH, ''),
        })
        return Sample(
            input=[ChatMessageUser(content=record['prompt'])],
            target=self._record_reference_image(record),
            metadata=metadata
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

                # fine-granular metrics
                category = meta.get('category')
                if category:
                    metric_name = f'{metric_name}_{category}'
                if isinstance(metric_score, dict):
                    for k, v in metric_score.items():
                        score.value[f'{metric_name}_{k}'] = self._score_to_float(v)
                else:
                    score.value[metric_name] = self._score_to_float(metric_score)
            except Exception as e:
                logger.error(f'Error calculating metric {metric}: {e}')
                score.value[metric_name] = 0
                score.metadata[metric_name] = f'error: {str(e)}'

        return score

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
            f'Metric {metric_name} requires a reference image. Provide one of: '
            f'{", ".join(IMAGE_PAIR_REFERENCE_KEYS)}.'
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
