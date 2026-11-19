# Copyright (c) Alibaba, Inc. and its affiliates.
from pathlib import Path
from typing import Any, Sequence

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import DataLoader, Dataset, LocalDataLoader, Sample, download_dataset_file
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    JudgeReview,
    Placement,
)
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser, ContentImage, ContentText
from evalscope.api.metric import MetricSelector, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import FileConstants, HubType, ScoreStatus, ScoringPolicy, Tags

from .checklists import DIM_TO_CHECKLIST, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .utils import GRADE_CONTRACT, SCORE_NAME, prepare_judge_image, reduce_grades

DATASET_FILE = 'qwen_image_bench_hf_v0518.jsonl'


@register_benchmark(
    BenchmarkMeta(
        name='qwen_image_bench',
        pretty_name='Qwen-Image-Bench',
        dataset_id='Qwen/Qwen-Image-Bench',
        hf_dataset_id='Qwen/Qwen-Image-Bench',
        hf_dataset_revision='d2493deb153b020cf169c7e3f57d15e4dd697038',
        paper_url='https://arxiv.org/abs/2605.28091',
        description="""
## Overview

Qwen-Image-Bench evaluates text-to-image models on professional creative tasks using the official Q-Judger.

## Task Description

- **Task Type**: Text-to-image generation and evaluation
- **Input**: A Chinese or English creative prompt
- **Output**: An image evaluated against applicable rubrics
- **Domain**: Image quality, aesthetics, alignment, real-world fidelity, and creative generation

## Key Features

- 1,000 bilingual prompts with 56 facets grouped into five dimensions.
- Supports both image generation and evaluation of existing images via a local JSONL manifest.

## Evaluation Notes

- Set `judge.models` to an OpenAI-compatible vision endpoint serving the official `Qwen/Qwen-Image-Bench`
  weights. Serve with thinking enabled and reasoning separated from the final JSON. Use temperature 0,
  top-k 1, top-p 1, repetition penalty 1.05, seed 42, and a 4,096-token generation limit for official settings.
- Set `extra_params.language` to `cn` (default) or `en`. Each run reports scores for the selected language.
- For existing images, use a local JSONL with `ID`, `dims_en`, `prompt` (or the selected official
  `prompt_cn`/`prompt_en` field), and an absolute `image_path`. Set `model_task='image_generation'`
  and leave `model` unset to score these images.
- Grades 0, 1, and 2 map to 0, 60, and 100. Inapplicable facets and failed requests are excluded from the
  corresponding averages. Each image is averaged across its scored dimensions before averaging images.
- The overall score, dimensions, and subdimensions report their respective valid image counts.

This configuration scores existing images against the English prompts.

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model_id='my-image-model',
    model_task='image_generation',
    datasets=['qwen_image_bench'],
    dataset_args={'qwen_image_bench': {
        'dataset_id': '/absolute/path/to/manifest.jsonl',
        'extra_params': {'language': 'en'},
    }},
    judge={'models': [{
        'model_id': 'Qwen/Qwen-Image-Bench',
        'api_url': 'http://127.0.0.1:8000/v1',
        'generation_config': {
            'temperature': 0,
            'top_p': 1,
            'seed': 42,
            'max_tokens': 4096,
            'extra_body': {
                'top_k': 1,
                'repetition_penalty': 1.05,
                'chat_template_kwargs': {'enable_thinking': True},
            },
        },
    }]},
))
```
""",
        tags=[Tags.TEXT_TO_IMAGE],
        metric_list=[SCORE_NAME],
        primary_metric=MetricSelector(name=SCORE_NAME, aggregation='mean', dimensions={'scope': 'overall'}),
        subset_list=['default'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        evaluation_version='v1.0',
        extra_params={
            'language': {
                'type': 'str',
                'description': 'Language of the generation prompts (cn or en).',
                'value': 'cn',
                'choices': ['cn', 'en'],
            },
        },
    )
)
class QwenImageBenchAdapter(Text2ImageAdapter):
    scoring_policy = ScoringPolicy.JUDGE_ONLY
    doc_eval_type = 'text2image'

    def load_subset(self, subset: str, data_loader: type[DataLoader]) -> Dataset:
        """Use a single dataset file, leaving the released example images on the hub."""
        if self.dataset_hub == HubType.LOCAL:
            return super().load_subset(subset, data_loader)
        path = download_dataset_file(
            self.dataset_id,
            DATASET_FILE,
            data_source=self.dataset_hub,
            revision=self.dataset_revision,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        )
        return LocalDataLoader(
            data_id_or_path=path,
            split=self.eval_split,
            subset=subset,
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
        ).load()

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        """Select the prompt language and retain only the fields needed to score the image."""
        prompt_field = f'prompt_{self.extra_params.get("language", "cn")}'
        prompt = record['prompt'] if 'prompt' in record else record[prompt_field]
        dimensions = dict.fromkeys(
            item.split('/', 1)[0].strip() for item in record['dims_en'].split(';') if item.strip()
        )
        if not dimensions or dimensions.keys() - DIM_TO_CHECKLIST.keys():
            raise ValueError(f'Invalid Qwen-Image-Bench dimensions for ID {record["ID"]}: {list(dimensions)}')
        return super().record_to_sample(
            {
                'id': record['ID'],
                'prompt': prompt,
                'dimensions': list(dimensions),
                'image_path': record.get('image_path', ''),
            }
        )

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        """Send the official checklist for each applicable dimension."""
        metadata = context.task_state.metadata or {}
        image_path = metadata.get(FileConstants.IMAGE_PATH, '')
        artifact_dir = None
        if self._task_config is not None:
            artifact_dir = Path(self._task_config.work_dir).resolve() / 'judge_images' / self.name
        try:
            judge_image = prepare_judge_image(image_path, artifact_dir)
        except (OSError, ValueError) as error:
            reason = f'Cannot load the generated image: {error}'
            score = Score(prediction=image_path, status=ScoreStatus.EXCLUDED, explanation=reason)
            return JudgeDefinition.skip(score, reason=reason)

        cases = [
            JudgeCase(
                case_id=dimension, output_contract=GRADE_CONTRACT, required=False, metadata={'dimension': dimension}
            )
            for dimension in metadata['dimensions']
        ]

        def request(
            case: JudgeCase, placement: Placement, completed: Sequence[CaseVerdict], judge_context: JudgeContext
        ) -> JudgeRequest:
            dimension = case.metadata['dimension']
            before_image, after_image = USER_PROMPT_TEMPLATE.split('<image>', 1)
            before_image = before_image.format(prompt=metadata['prompt'])
            after_image = after_image.format(level1_dim=dimension, format_checklist=DIM_TO_CHECKLIST[dimension])
            return JudgeRequest(
                messages=[
                    ChatMessageSystem(content=SYSTEM_PROMPT),
                    ChatMessageUser(
                        content=[
                            ContentText(text=before_image),
                            ContentImage(image=judge_image),
                            ContentText(text=after_image),
                        ]
                    ),
                ]
            )

        def finalize(score: Score, review: JudgeReview, judge_context: JudgeContext) -> Score:
            score.prediction = score.extracted_prediction = image_path
            if score.value:
                components = {key.partition(':')[2] or 'overall': value for key, value in score.value.items()}
                score.value.clear()
                self._record_metric_result(score, SCORE_NAME, components)
            return score

        return JudgeDefinition.workflow(
            cases=cases, request=request, reduce=reduce_grades, main_score_name=SCORE_NAME, finalize=finalize
        )
