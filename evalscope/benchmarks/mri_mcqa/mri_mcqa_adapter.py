from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

MRI-MCQA is a specialized benchmark composed of multiple-choice questions related to Magnetic Resonance Imaging (MRI). It evaluates AI models' understanding of MRI physics, protocols, image acquisition, and clinical applications.

## Task Description

- **Task Type**: Medical Imaging Knowledge Multiple-Choice QA
- **Input**: MRI-related question with multiple answer choices
- **Output**: Correct answer letter
- **Domain**: Medical imaging, MRI physics, radiology

## Key Features

- Specialized focus on MRI technology and applications
- Tests understanding of MRI physics and protocols
- Covers clinical MRI applications and sequences
- Designed for evaluating medical imaging AI systems
- Multiple-choice format for standardized evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on test split
- Simple accuracy metric
- No training split available
"""


@register_benchmark(
    BenchmarkMeta(
        name='mri_mcqa',
        pretty_name='MRI-MCQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MEDICAL],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/mri-mcqa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class MRIMCQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
