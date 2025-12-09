import ast
import numpy as np
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = r""" You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

    1. Text Processing:
    - Accurately recognize all text content in the PDF image without guessing or inferring.
    - Convert the recognized text into Markdown format.
    - Maintain the original document structure, including headings, paragraphs, lists, etc.

    2. Mathematical Formula Processing:
    - Convert all mathematical formulas to LaTeX format.
    - Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
    - Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

    3. Table Processing:
    - Convert tables to HTML format.
    - Wrap the entire table with <table> and </table>.

    4. Figure Handling:
    - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

    5. Output Format:
    - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
    - For complex layouts, try to maintain the original document's structure and format as closely as possible.

    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='omni_doc_bench',
        pretty_name='OmniDocBench',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=
        """OmniDocBench is an evaluation dataset for diverse document parsing in real-world scenarios, with the following characteristics:
- Diverse Document Types: The evaluation set contains 1355 PDF pages, covering 9 document types, 4 layout types and 3 language types. It has broad coverage including academic papers, financial reports, newspapers, textbooks, handwritten notes, etc.
- Rich Annotations: Contains location information for 15 block-level (text paragraphs, titles, tables, etc., over 20k in total) and 4 span-level (text lines, inline formulas, superscripts/subscripts, etc., over 80k in total) document elements, as well as recognition results for each element region (text annotations, LaTeX formula annotations, tables with both LaTeX and HTML annotations). OmniDocBench also provides reading order annotations for document components. Additionally, it includes various attribute labels at page and block levels, with 5 page attribute labels, 3 text attribute labels and 6 table attribute labels.
**The evaluation in EvalScope implements the `end2end` and `quick_match` methods from the official [OmniDocBench-v1.5 repository](https://github.com/opendatalab/OmniDocBench).**
""",  # noqa: E501
        dataset_id='evalscope/OmniDocBench_tsv',
        metric_list={
            'text_block': {
                'metric': ['Edit_dist', 'BLEU', 'METEOR']
            },
            'display_formula': {
                'metric': ['Edit_dist']
            },
            'table': {
                'metric': ['TEDS', 'Edit_dist']
            },
            'reading_order': {
                'metric': ['Edit_dist']
            }
        },
        eval_split='train',
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            'match_method': {
                'type': 'str',
                'description': 'Scoring match method used for evaluation.',
                'value': 'quick_match',
                'choices': ['quick_match', 'simple_match', 'no_split']
            }
        }
    )
)
class OmniDocBenchAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_aggregation_name = False
        self.match_method = self.extra_params.get('match_method', 'quick_match')

        check_import(
            module_name=['apted', 'distance', 'Levenshtein', 'lxml', 'bs4'],
            package=['apted', 'distance', 'Levenshtein', 'lxml', 'BeautifulSoup4'],
            raise_error=True,
            feature_name='OmniDocBench'
        )

    def record_to_sample(self, record) -> Sample:
        content_list: List[Content] = [ContentText(text=self.prompt_template)]
        content_list.append(ContentImage(image=f'data:image/png;base64,{record["image"]}'))

        return Sample(
            input=[ChatMessageUser(content=content_list)], target='', metadata=ast.literal_eval(record['answer'])
        )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        # Dummy implementation to comply with the interface

        score = Score(
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
        )

        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        from .end2end_eval import End2EndEvaluator

        if not sample_scores:
            return []

        predictions = [s.score.prediction for s in sample_scores]
        references = [s.sample_metadata for s in sample_scores]

        evaluator = End2EndEvaluator(
            prediction=predictions,
            reference=references,
            metrics=self.metric_list,
            match_method=self.match_method,
        )
        agg_results = evaluator.score()

        agg_scores = []
        for metric_name, agg_result in agg_results.items():
            if agg_result is not np.nan:
                agg_score = AggScore(
                    score=agg_result,
                    metric_name=metric_name,
                    num=len(sample_scores),
                )
                agg_scores.append(agg_score)

        return agg_scores
