from typing import Any, Dict, Iterator, List

from evalscope.perf.workloads.builtins.base import DatasetConverterBase
from evalscope.utils.io_utils import PIL_to_base64


class FlickrDatasetPlugin(DatasetConverterBase):
    """Read dataset and return prompt.
    Datasets: https://www.modelscope.cn/datasets/clip-benchmark/wds_flickr8k/files
    """

    def __init__(self, query_parameters):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        dataset = self.load_hub_dataset(dataset_id='clip-benchmark/wds_flickr8k', split='test')

        for item in dataset:
            pil_image = item['jpg']
            text = item['txt']
            base64_image = PIL_to_base64(pil_image, add_header=True)

            message = self.create_message(text=text, image_urls=base64_image)
            yield [message]
