import os
import time
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.utils.chat_service import ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage
from evalscope.utils.io_utils import OutputsStructure
from evalscope.utils.logger import get_logger
from ..local_model import LocalModel
from .base_adapter import BaseModelAdapter

logger = get_logger()


class T2IModelAdapter(BaseModelAdapter):
    """
    Text to image model adapter.
    """

    def __init__(self, model: LocalModel, **kwargs):
        super().__init__(model)

        self.task_config = kwargs.get('task_cfg', None)
        assert self.task_config is not None, 'Task config is required for T2I model adapter.'

        self.save_path = os.path.join(self.task_config.work_dir, OutputsStructure.PREDICTIONS_DIR,
                                      self.task_config.model_id, 'images')
        os.makedirs(self.save_path, exist_ok=True)

    def _model_generate(self, prompt, infer_cfg=None) -> List:
        """
        Generate images from the model.
        Args:
            prompt: The input prompt.
            infer_cfg: The inference configuration.
        Returns:
            The generated images.
        """
        infer_cfg = infer_cfg or {}

        sample = self.model(prompt=prompt, **infer_cfg).images
        return sample

    @torch.no_grad()
    def predict(self, inputs: List[dict], infer_cfg: Optional[dict] = None) -> List[dict]:
        """
        Args:
            inputs: The input data.
            infer_cfg: The inference configuration.
        Returns:
            The prediction results.
        """
        results = []
        for input_item in inputs:
            prompt = input_item['data'][0]
            image_id = input_item.get('id') or input_item.get('index')

            samples = self._model_generate(prompt, infer_cfg)

            choices_list = []
            for index, sample in enumerate(samples):
                image_file_path = os.path.join(self.save_path, f'{image_id}_{index}.jpeg')
                sample.save(image_file_path)
                logger.debug(f'Saved image to {image_file_path}')

                choice = ChatCompletionResponseChoice(
                    index=index, message=ChatMessage(content=image_file_path, role='assistant'), finish_reason='stop')
                choices_list.append(choice)

            res_d = ChatCompletionResponse(
                model=self.model_id, choices=choices_list, object='images.generations',
                created=int(time.time())).model_dump(exclude_unset=True)

            results.append(res_d)

        return results
