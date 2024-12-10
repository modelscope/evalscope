import base64
from io import BytesIO
from typing import Any, Dict, Iterator, List

from modelscope.msdatasets import MsDataset
from PIL import Image

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


def PIL_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


@register_dataset('flickr8k')
class FlickrDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    Datasets: https://www.modelscope.cn/datasets/clip-benchmark/wds_flickr8k/files
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        dataset = MsDataset.load('clip-benchmark/wds_flickr8k', split='test')

        for item in dataset:
            pil_image = item['jpg']
            base64_iamge = PIL_to_base64(pil_image)

            yield [{
                'role':
                'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Describe the image'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{base64_iamge}',
                        }
                    },
                ],
            }]