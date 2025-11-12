# flake8: noqa: E501
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64, compress_image_to_limit
from evalscope.utils.logger import get_logger

logger = get_logger()

# 定义提示模板
PROMPT_TEMPLATE = """{question}
\n\n\nLet's think step by step and give the final answer in curly braces,
like this: {{final answer}}"
"""

SUBSET_LIST = ['default']


@register_benchmark(
    BenchmarkMeta(
        name='zerobench',
        pretty_name='ZeroBench',
        dataset_id='evalscope/zerobench',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.MULTI_MODAL],
        description=
        'ZeroBench is a challenging visual reasoning benchmark for Large Multimodal Models (LMMs). It consists of a main set of 100 high-quality, manually curated questions covering numerous domains, reasoning types and image type. Questions in ZeroBench have been designed and calibrated to be beyond the capabilities of current frontier models. As such, none of the evaluated models achieves a non-zero pass@1 (with greedy decoding) or 5/5 reliability score.',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='zerobench',
        train_split='zerobench_subquestions',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class ZeroBenchAdapter(VisionLanguageAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._use_llm_judge = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question_text']
        content_list: List[Content] = [ContentText(text=self.prompt_template.format(question=question))]
        image = record['question_images_decoded']
        if len(image) > 0:
            for img in image:
                # Ensure image is under OpenAI's 10MB data-URI limit by compressing if needed
                processed_bytes, fmt = compress_image_to_limit(img['bytes'], 10_000_000)
                image_base64 = bytes_to_base64(processed_bytes, format=fmt, add_header=True)
                content_list.append(ContentImage(image=image_base64))

        metadata = {
            'question_id': record['question_id'],
            'question_images': record['question_images'],
            'image_attribution': record['image_attribution']
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)], target=record['question_answer'], metadata=metadata
        )
