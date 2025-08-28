import base64
import os
from typing import Optional

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
from evalscope.utils.io_utils import jsonl_to_list
from .default_data_adapter import DefaultDataAdapter

logger = get_logger()


class ImageEditAdapter(DefaultDataAdapter):
    """
    Support two methods:
    1. Inference using modelscope pipeline
    2. Load local inference jsonl file with key to corresponding prompt
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.local_file = self.extra_params.get('local_file', None)
        self.id_key = self.extra_params.get('id_key', FileConstants.ID)
        self.image_key = self.extra_params.get('image_key', FileConstants.IMAGE_PATH)
        self.local_data = self.load_local_file()

    def load_local_file(self) -> Optional[dict]:
        if not self.local_file:
            return None

        # Load file and check
        data_list = jsonl_to_list(self.local_file)
        data_dict = {}
        for record in data_list:
            if self.image_key not in record:
                raise ValueError(f"Image key '{self.image_key}' not found in record: {record}, file {self.local_file}")
            if self.id_key not in record:
                raise ValueError(f"ID key '{self.id_key}' not found in record: {record}, file {self.local_file}")

            image_path = record[self.image_key]
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(self.local_file), image_path)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file '{image_path}' not found.")

            data_dict[record[self.id_key]] = record
        return data_dict

    def get_image_path_from_id(self, image_id) -> Optional[str]:
        if not self.local_file:
            return None

        record = self.local_data.get(image_id)
        if not record:
            return None

        return record[self.image_key]

    def _post_process_samples(self):
        super()._post_process_samples()

        # Add local image path if exists
        for subset in self.test_dataset.keys():
            for sample in self.test_dataset[subset]:
                local_image_path = self.get_image_path_from_id(sample.metadata.get(FileConstants.ID))
                if local_image_path:
                    sample.metadata[FileConstants.IMAGE_PATH] = local_image_path

    def sample_filter(self, sample) -> bool:
        """
        Filter samples based on metadata availability.
        If local file is not available, all samples are considered valid.
        Otherwise, only samples with valid metadata and image path are kept.
        """
        if not self.local_data:
            return True
        else:
            if (not sample.metadata) or (not sample.metadata.get(FileConstants.IMAGE_PATH)):
                return False
            return True

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
            image_id = f'{sample.metadata.get(FileConstants.ID, sample.id)}_{sample.group_id}'
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
