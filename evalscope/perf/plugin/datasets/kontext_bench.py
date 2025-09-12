from typing import Any, Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.io_utils import PIL_to_base64


@register_dataset('kontext_bench')
class KontextDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    Datasets: https://modelscope.cn/datasets/black-forest-labs/kontext-bench/dataPeview
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        from modelscope.msdatasets import MsDataset
        dataset = MsDataset.load('black-forest-labs/kontext-bench', subset_name='default', split='test')

        for item in dataset:
            pil_image = item['image']
            text = item['instruction']
            base64_image = PIL_to_base64(pil_image, add_header=True)

            message = self.create_message(text=text, image_urls=base64_image)
            yield [message]
