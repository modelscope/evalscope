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

SUBSET_LIST = [
    'APP agent en', 'ASCII art classification en', 'key information extraction cn', 'key information extraction en',
    'key information mapping en', 'VQA with position en', 'chart parsing en', 'cognition VQA cn', 'cognition VQA en',
    'diagram QA en', 'document classification en', 'document parsing cn', 'document parsing en',
    'formula recognition cn', 'formula recognition en', 'handwritten answer extraction cn', 'math QA en',
    'full-page OCR cn', 'full-page OCR en', 'reasoning VQA en', 'reasoning VQA cn', 'fine-grained text recognition en',
    'science QA en', 'table parsing cn', 'table parsing en', 'text counting en', 'text grounding en',
    'text recognition en', 'text spotting en', 'text translation cn'
]


@register_benchmark(
    BenchmarkMeta(
        name='ocr_bench_v2',
        pretty_name='OCRBench-v2',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=
        'OCRBench v2 is a large-scale bilingual text-centric benchmark with currently the most comprehensive set of tasks (4x more tasks than the previous multi-scene benchmark OCRBench), the widest coverage of scenarios (31 diverse scenarios including street scene, receipt, formula, diagram, and so on), and thorough evaluation metrics, with a total of 10, 000 human-verified question-answering pairs and a high proportion of difficult samples.',  # noqa: E501
        dataset_id='evalscope/OCRBench_v2',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
    )
)
class OCRBenchV2Adapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_aggregation_name = False
        self.reformat_subset = True

        check_import(
            module_name=['apted', 'distance', 'Levenshtein', 'lxml', 'Polygon', 'zss'],
            package=['apted', 'distance', 'Levenshtein', 'lxml', 'Polygon3', 'zss'],
            raise_error=True,
            feature_name='OCRBench-v2 benchmark'
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:

        input_text = self.prompt_template.format(question=record['question'])
        content_list: List[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=json.dumps(record.get('answers'), ensure_ascii=False),  # answers is a list
            subset_key=record.get('type'),
            metadata={
                'question': record.get('question'),
                'answers': record.get('answers'),
                'eval': record.get('eval'),
                'dataset_name': record.get('dataset_name'),
                'type': record.get('type'),
                'bbox': record.get('bbox'),
                'bbox_list': record.get('bbox_list'),
                'content': record.get('content'),
            }
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from .utils import ocrbench_v2_process_results

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        doc = task_state.metadata
        pred = filtered_prediction

        score_value = ocrbench_v2_process_results(doc, pred)

        score.value = {'acc': score_value}
        return score

    def _on_generate_report_end(self, report: Report, output_dir, **kwargs):
        """
        Finalize the report generation process. Calculate the overall score.
        """
        # Lazy import to avoid changing top-level imports
        from evalscope.report import Category, weighted_average_from_subsets

        for metric in report.metrics:
            # Collect all subsets in a dictionary for easy access
            subset_dict: Dict[str, Subset] = {}
            for category in metric.categories:
                for subset in category.subsets:
                    subset_dict[subset.name] = subset

            # Define category groupings (per utils.ocrbench_v2_aggregate_accuracy)
            en_categories = {
                'text_recognition_en': ['text recognition en', 'fine-grained text recognition en', 'full-page OCR en'],
                'text_detection_en': ['text grounding en', 'VQA with position en'],
                'text_spotting_en': ['text spotting en'],
                'relationship_extraction_en': ['key information extraction en', 'key information mapping en'],
                'element_parsing_en':
                ['document parsing en', 'chart parsing en', 'table parsing en', 'formula recognition en'],
                'mathematical_calculation_en': ['math QA en', 'text counting en'],
                'visual_text_understanding_en': ['document classification en', 'cognition VQA en', 'diagram QA en'],
                'knowledge_reasoning_en':
                ['reasoning VQA en', 'science QA en', 'APP agent en', 'ASCII art classification en'],
            }
            cn_categories = {
                'text_recognition_cn': ['full-page OCR cn'],
                'relationship_extraction_cn': ['key information extraction cn', 'handwritten answer extraction cn'],
                'element_parsing_cn': ['document parsing cn', 'table parsing cn', 'formula recognition cn'],
                'visual_text_understanding_cn': ['cognition VQA cn'],
                'knowledge_reasoning_cn': ['reasoning VQA cn', 'text translation cn'],
            }

            # Compute per-category scores (unweighted average of member subsets)
            for cat_name, sub_names in en_categories.items():
                subset_dict[cat_name] = weighted_average_from_subsets(sub_names, subset_dict)
            for cat_name, sub_names in cn_categories.items():
                subset_dict[cat_name] = weighted_average_from_subsets(sub_names, subset_dict)

            # Compute EN (average of EN category scores) and CN (average of CN category scores)
            en_cat_names = list(en_categories.keys())
            cn_cat_names = list(cn_categories.keys())
            subset_dict['EN'] = weighted_average_from_subsets(en_cat_names, subset_dict)
            subset_dict['CN'] = weighted_average_from_subsets(cn_cat_names, subset_dict)

            # Compute OVERALL (average of EN and CN)
            subset_dict['OVERALL'] = weighted_average_from_subsets(['EN', 'CN'], subset_dict)

            # Prepare and append a dummy category to show all computed aggregates
            all_computed = en_cat_names + cn_cat_names + ['EN', 'CN', 'OVERALL']
            dummy_subsets = []
            for name in all_computed:
                if name in subset_dict:
                    s = subset_dict[name]
                    if s.num > 0:
                        s.name = name  # Ensure the name is set correctly
                        dummy_subsets.append(s)

            if dummy_subsets:
                metric.categories.append(Category(name='-', subsets=dummy_subsets))
