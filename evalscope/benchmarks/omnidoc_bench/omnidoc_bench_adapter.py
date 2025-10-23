import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.report.report import Report, Subset
from evalscope.utils.import_utils import check_import
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = []


@register_benchmark(
    BenchmarkMeta(
        name='omnidoc_bench',
        pretty_name='OmniDocBench',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=
        """OmniDocBench is an evaluation dataset for diverse document parsing in real-world scenarios, with the following characteristics:
- Diverse Document Types: The evaluation set contains 1355 PDF pages, covering 9 document types, 4 layout types and 3 language types. It has broad coverage including academic papers, financial reports, newspapers, textbooks, handwritten notes, etc.
- Rich Annotations: Contains location information for 15 block-level (text paragraphs, titles, tables, etc., over 20k in total) and 4 span-level (text lines, inline formulas, superscripts/subscripts, etc., over 80k in total) document elements, as well as recognition results for each element region (text annotations, LaTeX formula annotations, tables with both LaTeX and HTML annotations). OmniDocBench also provides reading order annotations for document components. Additionally, it includes various attribute labels at page and block levels, with 5 page attribute labels, 3 text attribute labels and 6 table attribute labels.
- High Annotation Quality: Through manual screening, intelligent annotation, manual annotation, full expert quality inspection and large model quality inspection, the data quality is relatively high.
**The evaluation in EvalScope implements the `end2end` and `quick_match` methods from the official [OmniDocBench-v1.5 repository](https://github.com/opendatalab/OmniDocBench).**
""",  # noqa: E501
        dataset_id='evalscope/OmniDocBench',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
    )
)
class OmniDocBenchAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_aggregation_name = False
        self.reformat_subset = True

        check_import(
            module_name=['apted', 'distance', 'editdistance', 'Levenshtein', 'lxml', 'Polygon', 'zss'],
            package=['apted', 'distance', 'editdistance', 'Levenshtein', 'lxml', 'Polygon3', 'zss'],
            raise_error=True,
            feature_name='OCRBench-v2 benchmark'
        )
