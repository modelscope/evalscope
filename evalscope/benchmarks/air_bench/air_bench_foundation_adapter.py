# Copyright (c) Alibaba, Inc. and its affiliates.
"""AIR-Bench Foundation track adapter (single-choice MCQ across 19 tasks).

Reference: Yang et al., "AIR-Bench: Benchmarking Large Audio-Language Models via
Generative Comprehension", ACL 2024 (https://arxiv.org/abs/2402.07729).
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from .utils import (
    FOUNDATION_SUBSET_TO_CATEGORY,
    HF_REPO_ID,
    audio_path_to_base64,
    download_air_bench,
    load_meta,
    normalise_audio_for_content,
    prepare_samples,
    resolve_audio_path,
)

logger = get_logger()

QUESTION_PROMPT = (
    'Choose the most suitable answer from options A, B, C, and D to respond '
    'the question in next line, you may only choose A or B or C or D.'
)

VALID_LETTERS = ('A', 'B', 'C', 'D')


@register_benchmark(
    BenchmarkMeta(
        name='air_bench_foundation',
        pretty_name='AIR-Bench-Foundation',
        dataset_id=HF_REPO_ID,
        paper_url='https://aclanthology.org/2024.acl-long.109/',
        tags=[Tags.AUDIO, Tags.MULTIPLE_CHOICE, Tags.KNOWLEDGE],
        description="""
## Overview

AIR-Bench Foundation is the discriminative half of [AIR-Bench](https://arxiv.org/abs/2402.07729) (Audio InstRuction Benchmark, ACL 2024 main conference) — the first instruction-following benchmark for large audio-language models (LALMs), covering **human speech, natural sounds and music**. The Foundation track contains roughly 25k single-choice questions spanning 19 logical tasks across three audio categories.

## Task Description

- **Task Type**: Single-choice question answering grounded on an audio clip.
- **Input**: One audio clip + a question with up to four candidate answers (A/B/C/D).
- **Output**: A single letter chosen from the provided options.

## Categories (19 tasks / 25 source-dataset subsets)

- **Speech** (11 dirs / 9 tasks): speech grounding, language ID, gender, emotion (IEMOCAP+MELD), age, speech entity recognition, intent classification, speaker counting, synthesized-voice detection.
- **Sound** (6 dirs / 4 tasks): audio grounding, vocal sound classification, acoustic scene classification (CochlScene+TUT2017), sound QA (avqa+clothoaqa).
- **Music** (8 dirs / 6 tasks): instruments, genre, MIDI pitch, MIDI velocity, music QA, music emotion.

## Prompt Template (matches official `Inference_Foundation.py`)

```text
Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D.
{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
```

## Dataset Access

- The dataset is hosted on ModelScope: [`evalscope/AIR-Bench-Dataset`](https://modelscope.cn/datasets/evalscope/AIR-Bench-Dataset). It uses an *audiofolder + JSON metadata* layout. evalscope downloads it lazily via `modelscope.dataset_snapshot_download` on first run; the full release is ~49 GB, so it is recommended to limit which subsets are pulled via `extra_params`.
- If the dataset is already on disk, pass `dataset_args={'air_bench_foundation': {'local_path': '/path/to/AIR-Bench-Dataset'}}`; the local root should contain `Foundation/`.
- Some Foundation samples are FLAC. For OpenAI-compatible audio input evalscope converts them to cached WAV files, which requires either `soundfile` (`pip install "evalscope[air_bench]"`) or a working `ffmpeg` binary.

## Evaluation Notes

- Metric: **accuracy** (per source-dataset subset, plus per-category aggregation).
- Default prompt follows the official `Inference_Foundation.py` formatting so existing AIR-Bench leaderboard numbers can be reproduced.
- Set `extra_params={'subsets': [...]}` to limit to a subset of the 25 source directories — useful for partial downloads.
""",  # noqa: E501
        subset_list=list(FOUNDATION_SUBSET_TO_CATEGORY.keys()),
        eval_split='test',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        prompt_template=QUESTION_PROMPT,
        extra_params={
            'subsets': {
                'type': 'list',
                'description': 'Optional list of Foundation source-dataset directories to evaluate. '
                'Defaults to all 25 directories. Useful when only a subset has been downloaded locally.',  # noqa: E501
                'value': None,
            },
        },
    )
)
class AIRBenchFoundationAdapter(VisionLanguageAdapter):
    """Adapter for AIR-Bench Foundation single-choice MCQ tasks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = False
        self.add_aggregation_name = False
        self.category_map = FOUNDATION_SUBSET_TO_CATEGORY
        self._track_root: Optional[str] = None
        self._audio_cache_dir = ''

    # ------------------------------------------------------------------
    # Dataset loading (overridden — AIR-Bench is a soundfolder + JSON layout
    # that ``datasets.load_dataset`` cannot consume directly).
    # ------------------------------------------------------------------
    def load(self) -> Tuple[DatasetDict, None]:
        requested_subsets = self.extra_params.get('subsets') or list(self.subset_list)
        unknown = [s for s in requested_subsets if s not in FOUNDATION_SUBSET_TO_CATEGORY]
        if unknown:
            raise ValueError(
                f'Unknown AIR-Bench Foundation subset(s): {unknown}. '
                f'Valid choices: {sorted(FOUNDATION_SUBSET_TO_CATEGORY)}.'
            )

        track_root = download_air_bench(
            track='Foundation',
            dataset_id=self.dataset_id,
            cache_dir=self.dataset_dir,
            subset_dirs=requested_subsets,
        )
        self._track_root = track_root
        self._audio_cache_dir = os.path.join(self.dataset_dir, 'air_bench_converted_audio')
        records = load_meta(track_root, 'Foundation')

        wanted = set(requested_subsets)
        per_subset_samples: Dict[str, List[Sample]] = {s: [] for s in requested_subsets}
        skipped_missing_audio = 0

        for record in records:
            folder = f"{record['task_name']}_{record['dataset_name']}"
            if folder not in wanted:
                continue
            sample = self._record_to_sample_with_root(record, track_root)
            if sample is None:
                skipped_missing_audio += 1
                continue
            per_subset_samples[folder].append(sample)

        if skipped_missing_audio:
            logger.warning(
                f'AIR-Bench Foundation: skipped {skipped_missing_audio} samples whose audio files '
                f'were missing on disk (likely partial download).'
            )

        dataset_dict = DatasetDict({
            k: prepare_samples(
                v,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
                seed=self.seed,
                name=f'air_bench_foundation/{k}',
            )
            for k, v in per_subset_samples.items()
        })
        return dataset_dict, None

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # NOTE: this default override is kept for API compatibility — actual
        # sample construction happens in ``_record_to_sample_with_root`` so we
        # have access to the resolved track root for path joining.
        if self._track_root is None:
            raise RuntimeError(
                '`_track_root` is not initialised; AIR-Bench samples must be '
                'constructed via `load()`.'
            )
        sample = self._record_to_sample_with_root(record, self._track_root)
        if sample is None:
            raise FileNotFoundError(
                f"Audio file missing for AIR-Bench Foundation record uniq_id={record.get('uniq_id')}."
            )
        return sample

    def _record_to_sample_with_root(self, record: Dict[str, Any], track_root: str) -> Optional[Sample]:
        task_name = record['task_name']
        dataset_name = record['dataset_name']

        audio_path, audio_format = resolve_audio_path(
            track_root=track_root,
            track='Foundation',
            task_name=task_name,
            dataset_name=dataset_name,
            rel_path=record['path'],
        )
        if not os.path.exists(audio_path):
            return None
        audio_path, audio_format = normalise_audio_for_content(
            audio_path,
            audio_format,
            cache_dir=self._audio_cache_dir,
        )
        audio_b64 = audio_path_to_base64(audio_path, audio_format)

        question = record['question']
        choice_a = record.get('choice_a')
        choice_b = record.get('choice_b')
        choice_c = record.get('choice_c')
        choice_d = record.get('choice_d')

        choices_block = (f'A. {choice_a}\nB. {choice_b}\n'
                         f'C. {choice_c}\nD. {choice_d}')
        instruction = f'{QUESTION_PROMPT}\n{question}\n{choices_block}'

        target_letter = self._answer_to_letter(
            answer_gt=record['answer_gt'],
            choices=(choice_a, choice_b, choice_c, choice_d),
        )

        folder = f'{task_name}_{dataset_name}'
        category = FOUNDATION_SUBSET_TO_CATEGORY[folder]

        return Sample(
            input=[
                ChatMessageUser(
                    content=[
                        ContentAudio(audio=audio_b64, format=audio_format),
                        ContentText(text=instruction),
                    ]
                )
            ],
            target=target_letter,
            subset_key=folder,
            metadata={
                'uniq_id': record.get('uniq_id'),
                'task_name': task_name,
                'dataset_name': dataset_name,
                'category': category,
                'answer_gt_text': record['answer_gt'],
                'choices': {
                    'A': choice_a,
                    'B': choice_b,
                    'C': choice_c,
                    'D': choice_d
                },
            },
        )

    @staticmethod
    def _answer_to_letter(answer_gt: Any, choices: Tuple[Any, ...]) -> str:
        """Return the letter (A/B/C/D) matching ``answer_gt`` against ``choices``.

        AIR-Bench's `answer_gt` field stores either the letter directly or the
        full text of the correct option. We normalise both forms.
        """
        if isinstance(answer_gt, str):
            stripped = answer_gt.strip().rstrip('.').strip()
            if len(stripped) == 1 and stripped.upper() in VALID_LETTERS:
                return stripped.upper()
            for letter, choice_text in zip(VALID_LETTERS, choices):
                if choice_text is None:
                    continue
                if str(choice_text).strip() == stripped:
                    return letter
        # Fall back to a sentinel that can never match a model letter; the
        # sample will simply be scored 0.
        return ''

    # ------------------------------------------------------------------
    # Answer extraction & scoring
    # ------------------------------------------------------------------
    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        if not prediction:
            return ''
        # Look for the first standalone A/B/C/D, optionally prefixed by markers
        # like ``ANSWER:`` or wrapped in parentheses/brackets.
        match = re.search(r'(?i)answer\s*[:\-]?\s*([ABCD])', prediction)
        if match:
            return match.group(1).upper()
        match = re.search(r'\b([ABCD])\b', prediction)
        if match:
            return match.group(1).upper()
        return prediction.strip()[:1].upper()

    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        is_correct = (
            isinstance(filtered_prediction, str) and isinstance(reference, str) and reference != ''
            and filtered_prediction.upper() == reference.upper()
        )
        score.value = {'acc': 1.0 if is_correct else 0.0}
        score.main_score_name = 'acc'
        return score
