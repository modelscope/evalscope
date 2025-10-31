# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy
import json
import re
from pathlib import Path
from typing import List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags, JSON_PATTERNS
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = ["default"]
METRIC_LIST = ['anls', 'exact_match']
NONE_ANSWER = "NaN"
COLUMN_METRIC_DICT = {
    "compound name": METRIC_LIST[0],
    "metal source": METRIC_LIST[0],
    "metal amount": METRIC_LIST[0],
    "linker": METRIC_LIST[0],
    "linker amount": METRIC_LIST[0],
    "modulator": METRIC_LIST[0],
    "modulator amount or volume": METRIC_LIST[0],
    "solvent": METRIC_LIST[0],
    "solvent volume": METRIC_LIST[0],
    "reaction temperature": METRIC_LIST[0],
    "reaction time": METRIC_LIST[0]
}


INFORMATION_EXTRACTION_PROMPT = r"""You are an advanced information extraction system specialized in MOF (Metal-Organic Framework) synthesis documentation. Your task is to extract precise experimental parameters from scientific texts with high accuracy and consistency.
Schema:

```json
{
  "compound name": "The name or identifier of the synthesized MOF compound",
  "metal source": "The precursor compound that supplies metal ions",
  "metal amount": "The quantity of the metal precursor used",
  "linker": "The organic ligand that connects metal nodes to form the MOF framework",
  "linker amount": "The quantity of the organic ligand used",
  "modulator": "A modulating agent",
  "modulator amount or volume": "The quantity of the modulator used",
  "solvent": "The solvent system employed in the reaction",
  "solvent volume": "The volume ratio and total amount of solvent",
  "reaction temperature": "The temperature condition at which the reaction proceeds",
  "reaction time": "The duration of the reaction"
}
```

EXTRACTION PROTOCOL:

Comprehensive Coverage: Extract at least one complete data set per text. Focus on extracting coherent, internally consistent parameter sets rather than forcing multiple incomplete entries.

Source Verification: For each attribute, provide the exact source sentence(s) where the information was extracted, using natural sentence boundaries (periods, semicolons, exclamation marks, question marks).

Null Value Handling: Assign "NaN" exclusively for attributes that are genuinely absent from the text. Do not infer or assume values.

Schema Compliance: Output must include all schema attributes for each compound. Different compounds may share identical values when explicitly stated.

Quality Control: Discard any data set where the majority of attributes are empty ("NaN") or cannot be reliably extracted.

Textual Fidelity: Extract and use only the exact wording from the source text. No paraphrasing, correction, or interpretation is permitted under any circumstances.

OUTPUT REQUIREMENT:
Maintain strict adherence to the original text's terminology, units, and expressions. Preserve all numerical values, chemical names, and experimental conditions exactly as written in the source material.

RESPONSE FORMAT:
You must present your final extraction results within a markdown JSON code block
"""

@register_benchmark(
    BenchmarkMeta(
        name='internal_mof_information_extraction',
        pretty_name='IMOFIE',
        description='Internal dataset with MOF information extraction task',
        tags=[Tags.MULTI_MODAL, Tags.CUSTOM],
        dataset_id='/app/custom_eval/internal/MOF-extraction',
        subset_list=SUBSET_LIST,
        metric_list=METRIC_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=INFORMATION_EXTRACTION_PROMPT,
        extra_params={
            'use_image': True,  # Whether to use image input, if False, use json text alternative image content.
            'image_concat': False, # Whether to concat pages into one big image, available only if `use_image` is True
            'column_metric_dict': COLUMN_METRIC_DICT,
            'none_answer': NONE_ANSWER   # TODO: use it
        }
    )
)
class IMOFIEAdapter(DefaultDataAdapter):
    """
    something named internal MOF information extraction dataset adapter
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.use_image = self.extra_params.get('use_image', True)  # whether to treat pdf as images
        self.image_concat = self.extra_params.get('image_concat', False)
        self.column_metric_dict = self.extra_params.get('column_metric_dict', {})

        self.dataset_path = Path(self.dataset_id)

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record) -> Sample:
        answer = record['mof_gt'][0]
        input_text = INFORMATION_EXTRACTION_PROMPT
        content_list: List[Content] = [ContentText(text=input_text)]

        if self.use_image:
            file_path = self.dataset_path / "pdf" / (record['file name short']+".pdf")
            image_base64 = self._pdf_to_base64(file_path)
            for image in image_base64:
                content_list.append(ContentImage(image=image))
        else:
            alt_text = record['content']
            content_list.append(ContentText(text=alt_text))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=json.dumps(answer),
            metadata={
                'id': record.get('file ID', 'unknown'),
                'name': record.get('file name short', 'unknown'),
                'doi': record.get('doi', 'unknown'),
                'mof_gt': json.dumps(answer)
            }
        )

    def _pdf_to_base64(self, file_path):
        import base64
        import fitz
        document = fitz.open(file_path)
        # self.image_concat  # TODO: 拼接逻辑
        images = []
        for page in document:
            pix_image = page.get_pixmap().tobytes("png")
            base64_image = base64.b64encode(pix_image)
            images.append("data:image/png;base64," + str(base64_image, encoding="utf8"))
        document.close()
        return images

    def extract_answer(self, prediction, task_state):
        """extract first json code block
        """
        answer = "{}"
        if not prediction or not isinstance(prediction, str):
            return answer
        for pattern in JSON_PATTERNS:
            matches = re.findall(pattern, prediction, re.DOTALL)
            if matches:
                answer = matches[0].strip()
                try:
                    json.loads(answer)
                    break
                except json.JSONDecodeError as e:
                    continue
        return answer
    
    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        """总而言之我的目标是
        这里涉及11个key，分为exact match（acc）和编辑距离, NaN是一个特殊情况。
        所以规则是：如果gt的某个key是NaN，就exact match；否则按照COLUMN_METRIC_DICT来
        """
        from .utils import calculate_metrics

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            metadata=deepcopy(self.extra_params)
        )

        pred = filtered_prediction.strip()
        gt_ans = json.loads(reference)

        try:
            pred = json.loads(pred)
            data = {
                "target": gt_ans,
                "predictions": pred,
            }
            metrics = calculate_metrics(data, self.column_metric_dict)  # returns dict

            score.value = metrics
            score.explanation = f"如果gt的某个key是NaN，就exact match；否则按照COLUMN_METRIC_DICT来。COLUMN_METRIC_DICT:{self.column_metric_dict}"
        except Exception as e:
            # Handle evaluation errors
            score.value = calculate_metrics(None, self.column_metric_dict)
            score.explanation = f'Evaluation failed: {str(e)}'
            score.metadata.update({'error': str(e)})
        score.main_score_name = 'ave'

        return score

