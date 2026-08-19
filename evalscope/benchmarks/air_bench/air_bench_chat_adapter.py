# Copyright (c) Alibaba, Inc. and its affiliates.
"""AIR-Bench Chat track adapter (open-ended audio QA scored by GPT-4 judge).

Reference: Yang et al., "AIR-Bench: Benchmarking Large Audio-Language Models via
Generative Comprehension", ACL 2024 (https://arxiv.org/abs/2402.07729).

Notes on judge fidelity
-----------------------
The official AIR-Bench leaderboard fixes the judge model to ``gpt-4-0125-preview``.
If that exact snapshot is unavailable, use an available GPT-4-class judge and
expect absolute scores to drift slightly versus the published numbers. This is a
known property of the benchmark, not an implementation difference.
"""

import os
import re
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import AudioLanguageAdapter, BenchmarkMeta
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeRequest, OutputContract, Placement, ReducedVerdict
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser, ContentAudio, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger
from .utils import (
    CHAT_TASK_TO_CATEGORY,
    HF_REPO_ID,
    audio_path_to_base64,
    download_air_bench,
    load_meta,
    normalise_audio_for_content,
    prepare_samples,
    resolve_audio_path,
)

logger = get_logger()


class PairRating(BaseModel):
    """Both assistants' 1-10 ratings from one judge pass."""
    assistant1: float = Field(ge=1.0, le=10.0)
    assistant2: float = Field(ge=1.0, le=10.0)


PAIR_CONTRACT = OutputContract(schema_model=PairRating)

JUDGE_SYSTEM_PROMPT = ('You are a helpful and precise assistant for checking the quality of the answer.')

JUDGE_TEMPLATE = """[Detailed Audio Description]
{meta_info}
[Question]
{question}
[The Start of Assistant 1s Answer]
{assistant1}
[The End of Assistant 1s Answer]
[The Start of Assistant 2s Answer]
{assistant2}
[The End of Assistant 2s Answer]
[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.
Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. """  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='air_bench_chat',
        pretty_name='AIR-Bench-Chat',
        dataset_id=HF_REPO_ID,
        paper_url='https://aclanthology.org/2024.acl-long.109/',
        tags=[Tags.AUDIO, Tags.QA, Tags.INSTRUCTION_FOLLOWING],
        description="""
## Overview

AIR-Bench Chat is the generative half of [AIR-Bench](https://arxiv.org/abs/2402.07729) (Audio InstRuction Benchmark, ACL 2024 main conference) — the first instruction-following benchmark for large audio-language models (LALMs), covering **human speech, natural sounds and music**. It contains roughly 2k open-ended audio QA pairs covering speech, sound, music and mixed-audio scenes; responses are graded by a GPT-4 judge against a reference answer.

## Task Description

- **Task Type**: Open-ended audio question answering.
- **Input**: An audio clip plus a free-form question.
- **Output**: A textual answer evaluated against the reference response.
- **Modalities**: Audio (human speech, natural sounds, music) + text.

## Key Features

- ~2k open-ended audio QA pairs across speech, sound, music and mixed-audio scenes; the generative half of AIR-Bench (ACL 2024).
- 8 Chat tasks aggregated by the official `cal_score.py` into 5 reported categories: `speech` (`speech_QA`, `speech_dialogue_QA`), `sound` (`sound_QA`, `sound_generation_QA`), `music` (`music_QA`, `music_generation_analysis_QA`), `speech_and_sound` (`speech_and_sound_QA`), `speech_and_music` (`speech_and_music_QA`). The paper's Mixed-audio = mean(speech_and_sound, speech_and_music).
- Position bias is removed by judging every sample twice with reference/prediction order swapped, then averaging (disable via `extra_params={'do_swap': False}` to halve judge cost).
- Hosted on ModelScope ([`evalscope/AIR-Bench-Dataset`](https://modelscope.cn/datasets/evalscope/AIR-Bench-Dataset)) in an audiofolder + JSON layout; the full release is ~49 GB, so limit tasks via `extra_params={'tasks': [...]}` for partial runs.

## Evaluation Notes

- Metrics: `judge_score` is the model's mean judge score; `win_rate` records how often the model strictly beats the reference.
- The judge LLM receives the question, the textual audio description (`meta_info`), the reference answer (`answer_gt`), and the model's response, and outputs two integer scores in `[1, 10]`. Use a judge that supports long contexts, since `meta_info` may exceed 4k tokens for dialogue tasks.
- The official leaderboard uses `gpt-4-0125-preview`. If that exact snapshot is unavailable, use an available GPT-4-class judge; absolute scores can drift versus the published numbers because the judge model changed.
- If the dataset is already on disk, pass `dataset_args={'air_bench_chat': {'local_path': '/path/to/AIR-Bench-Dataset'}}`; the local root should contain `Chat/`.
""",  # noqa: E501
        subset_list=list(CHAT_TASK_TO_CATEGORY.keys()),
        eval_split='test',
        metric_list=['judge_score', 'win_rate'],
        primary_metric='judge_score',
        few_shot_num=0,
        train_split=None,
        prompt_template='{question}',
        extra_params={
            'tasks': {
                'type': 'list',
                'description': 'Optional list of Chat task names to evaluate (subset of '
                f'{sorted(CHAT_TASK_TO_CATEGORY)}). Defaults to all tasks.',
                'value': None,
            },
            'do_swap': {
                'type': 'bool',
                'description': 'When True (default), each sample is judged twice with the order of '
                'reference vs. prediction swapped, then scores are averaged. Disable '
                'to halve judge cost at the price of position bias.',
                'value': True,
            },
        },
    )
)
class AIRBenchChatAdapter(AudioLanguageAdapter):
    """Adapter for AIR-Bench Chat open-ended audio QA tasks."""
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    # Per-sample folder layout for audio files. Distinct from Foundation since
    # Chat pre-merges some categories.
    TASK_DATASET_TO_FOLDER: Dict[Tuple[str, str], str] = {
        ('speech_QA', 'common_voice_en'): 'speech_QA_common_voice_en',
        ('speech_QA', 'iemocap'): 'speech_QA_iemocap',
        ('speech_dialogue_QA', 'fisher'): 'speech_dialogue_QA_fisher',
        ('speech_dialogue_QA', 'spokenwoz'): 'speech_dialogue_QA_spokenwoz',
        ('sound_QA', 'clotho'): 'sound_QA_clotho',
        ('sound_generation_QA', 'clotho'): 'sound_generation_QA_clotho',
        ('music_QA', 'musiccaps'): 'music_QA_musiccaps',
        ('music_generation_analysis_QA', 'musiccaps'): 'music_generation_analysis_QA_musiccaps',
        ('speech_and_sound_QA', 'audiocaps_cv'): 'speech_and_sound_QA_audiocaps_cv',
        ('speech_and_music_QA', 'musiccaps_cv'): 'speech_and_music_QA_musiccaps_cv',
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category_map = CHAT_TASK_TO_CATEGORY
        self._track_root: Optional[str] = None
        self._audio_cache_dir = ''

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def load(self) -> Tuple[DatasetDict, None]:
        requested_tasks = self.extra_params.get('tasks') or list(self.subset_list)
        unknown = [t for t in requested_tasks if t not in CHAT_TASK_TO_CATEGORY]
        if unknown:
            raise ValueError(
                f'Unknown AIR-Bench Chat task(s): {unknown}. '
                f'Valid choices: {sorted(CHAT_TASK_TO_CATEGORY)}.'
            )

        # Pull the directories that match the requested tasks.
        relevant_folders = sorted({
            folder
            for (task, _ds), folder in self.TASK_DATASET_TO_FOLDER.items()
            if task in requested_tasks
        })

        track_root = download_air_bench(
            track='Chat',
            dataset_id=self.dataset_id,
            cache_dir=self.dataset_dir,
            subset_dirs=relevant_folders,
        )
        self._track_root = track_root
        self._audio_cache_dir = os.path.join(self.dataset_dir, 'air_bench_converted_audio')
        records = load_meta(track_root, 'Chat')

        wanted = set(requested_tasks)
        per_subset_samples: Dict[str, List[Sample]] = {t: [] for t in requested_tasks}
        skipped_missing_audio = 0

        for record in records:
            task_name = record.get('task_name')
            if task_name not in wanted:
                continue
            sample = self._record_to_sample_with_root(record, track_root)
            if sample is None:
                skipped_missing_audio += 1
                continue
            per_subset_samples[task_name].append(sample)

        if skipped_missing_audio:
            logger.warning(
                f'AIR-Bench Chat: skipped {skipped_missing_audio} samples whose audio files '
                f'were missing on disk (likely partial download).'
            )

        dataset_dict = DatasetDict({
            k: prepare_samples(
                v,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
                seed=self.seed,
                name=f'air_bench_chat/{k}',
            )
            for k, v in per_subset_samples.items()
        })
        return dataset_dict, None

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        if self._track_root is None:
            raise RuntimeError(
                '`_track_root` is not initialised; AIR-Bench samples must be '
                'constructed via `load()`.'
            )
        sample = self._record_to_sample_with_root(record, self._track_root)
        if sample is None:
            raise FileNotFoundError(f"Audio file missing for AIR-Bench Chat record uniq_id={record.get('uniq_id')}.")
        return sample

    def _record_to_sample_with_root(self, record: Dict[str, Any], track_root: str) -> Optional[Sample]:
        task_name = record['task_name']
        dataset_name = record['dataset_name']
        question = record['question']

        folder = self.TASK_DATASET_TO_FOLDER.get((task_name, dataset_name))
        if folder is None:
            logger.warning(f'AIR-Bench Chat: unknown task/dataset combo ({task_name}/{dataset_name}); skipping.')
            return None

        audio_path, audio_format = resolve_audio_path(
            track_root=track_root,
            track='Chat',
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

        category = CHAT_TASK_TO_CATEGORY[task_name]

        return Sample(
            input=[
                ChatMessageUser(
                    content=[
                        ContentAudio(audio=audio_b64, format=audio_format),
                        ContentText(text=question),
                    ]
                )
            ],
            target=record['answer_gt'],
            subset_key=task_name,
            metadata={
                'uniq_id': record.get('uniq_id'),
                'task_name': task_name,
                'dataset_name': dataset_name,
                'category': category,
                'meta_info': record.get('meta_info', ''),
                'question': question,
            },
        )

    # ------------------------------------------------------------------
    # Scoring (LLM judge with optional position swap)
    # ------------------------------------------------------------------
    @property
    def judge_position_swap(self) -> bool:
        # Official cal_score.py judges each sample twice with the order swapped.
        return bool(self.extra_params.get('do_swap', True))

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        return [JudgeCase(case_id='pair', output_contract=PAIR_CONTRACT)]

    def build_judge_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        metadata = context.task_state.metadata or {}
        # ``assistant1`` is the reference on the original pass and the prediction on the swapped one.
        reference_first = placement is Placement.ORIGINAL
        prompt = JUDGE_TEMPLATE.format(
            meta_info=metadata.get('meta_info', ''),
            question=metadata.get('question') or context.task_state.input_text,
            assistant1=context.reference if reference_first else context.filtered_prediction,
            assistant2=context.filtered_prediction if reference_first else context.reference,
        )
        prompt += case.output_contract.instruction()
        return JudgeRequest(messages=[ChatMessageSystem(content=JUDGE_SYSTEM_PROMPT), ChatMessageUser(content=prompt)])

    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        placements = case_verdicts[0].placements
        if placements:
            # Both sides survived, or the executor would not have produced a verdict at all.
            original, swapped = placements['original'], placements['swapped']
            pred_scores = [original.assistant2, swapped.assistant1]
            ref_scores = [original.assistant1, swapped.assistant2]
        else:
            verdict = case_verdicts[0].value
            pred_scores = [verdict.assistant2]
            ref_scores = [verdict.assistant1]

        mean_pred = sum(pred_scores) / len(pred_scores)
        mean_ref = sum(ref_scores) / len(ref_scores)
        return ReducedVerdict(
            value={
                'judge_score': mean_pred,
                'win_rate': 1.0 if mean_pred > mean_ref else 0.0,
            },
            metadata={
                'reference_score': mean_ref,
                'pred_scores_per_pass': pred_scores,
                'reference_scores_per_pass': ref_scores,
            },
        )

    def finalize_judge_score(self, review, context) -> Score:
        score = super().finalize_judge_score(review, context)
        score.main_score_name = 'judge_score'
        return score
