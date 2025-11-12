import os
from typing import Optional

from evalscope.constants import EvalType, FileConstants
from evalscope.utils import get_logger
from evalscope.utils.function_utils import thread_safe
from evalscope.utils.io_utils import jsonl_to_list
from .text2image_adapter import Text2ImageAdapter

logger = get_logger()


class ImageEditAdapter(Text2ImageAdapter):
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
            sample_id = sample.metadata.get(FileConstants.ID)
            if (not sample_id) or (not self.get_image_path_from_id(sample_id)):
                return False
            return True
