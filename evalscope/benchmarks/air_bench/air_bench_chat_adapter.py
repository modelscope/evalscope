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
Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space."""  # noqa: E501


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

## Categories (8 tasks → 5 reported categories)

The 8 Chat tasks are aggregated by the official `cal_score.py` into five categories:

- `speech`: `speech_QA`, `speech_dialogue_QA`
- `sound`: `sound_QA`, `sound_generation_QA`
- `music`: `music_QA`, `music_generation_analysis_QA`
- `speech_and_sound`: `speech_and_sound_QA`
- `speech_and_music`: `speech_and_music_QA`

The paper's **Mixed-audio = mean(speech_and_sound, speech_and_music)**.

## Dataset Access

- The dataset is hosted on ModelScope: [`evalscope/AIR-Bench-Dataset`](https://modelscope.cn/datasets/evalscope/AIR-Bench-Dataset). It uses an *audiofolder + JSON metadata* layout. evalscope downloads it lazily via `modelscope.dataset_snapshot_download` on first run; the full release is ~49 GB, so it is recommended to limit which tasks are pulled via `extra_params`.
- If the dataset is already on disk, pass `dataset_args={'air_bench_chat': {'local_path': '/path/to/AIR-Bench-Dataset'}}`; the local root should contain `Chat/`.

## Evaluation Protocol

- The judge LLM (default: GPT-4) receives the question, the textual audio description (`meta_info` from the dataset), the reference answer (`answer_gt`), and the model's response. It outputs a single line with two integer scores in `[1, 10]`.
- To remove position bias, every sample is judged twice with the order of reference and prediction swapped, then averaged. This mirrors `cal_score.py` in the official repository — disable it via `extra_params={'do_swap': False}` to halve judge cost.
- Reported metric `gpt_score` is the model's mean judge score; `win_rate` records how often the model strictly beats the reference.

```{warning}
The official leaderboard uses `gpt-4-0125-preview` as the judge model. If that exact snapshot is unavailable, use an available GPT-4-class judge; absolute scores can drift versus the published numbers because the judge model changed.
```

## Implementation Notes

- The judge model is selected via `--judge-model-args`; ensure the model id supports long contexts (`meta_info` may exceed 4k tokens for dialogue tasks).
- Set `extra_params={'tasks': [...]}` to evaluate only specific Chat task names — useful for partial runs.
""",  # noqa: E501
        subset_list=list(CHAT_TASK_TO_CATEGORY.keys()),
        eval_split='test',
        metric_list=['gpt_score', 'win_rate'],
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
class AIRBenchChatAdapter(VisionLanguageAdapter):
    """Adapter for AIR-Bench Chat open-ended audio QA tasks."""

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
        self._use_llm_judge = True
        self.add_aggregation_name = False
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
    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        meta_info = task_state.metadata.get('meta_info', '')
        question = task_state.metadata.get('question') or task_state.input_text
        do_swap = bool(self.extra_params.get('do_swap', True))

        # Pass 1: reference as Assistant 1, prediction as Assistant 2.
        prompt_a = JUDGE_TEMPLATE.format(
            meta_info=meta_info,
            question=question,
            assistant1=reference,
            assistant2=filtered_prediction,
        )
        ref_score_1, pred_score_1, raw_1 = self._judge_pair(prompt_a)

        scores_ref: List[float] = []
        scores_pred: List[float] = []
        if ref_score_1 is not None and pred_score_1 is not None:
            scores_ref.append(ref_score_1)
            scores_pred.append(pred_score_1)

        raw_responses = [raw_1]

        if do_swap:
            # Pass 2: prediction as Assistant 1, reference as Assistant 2.
            prompt_b = JUDGE_TEMPLATE.format(
                meta_info=meta_info,
                question=question,
                assistant1=filtered_prediction,
                assistant2=reference,
            )
            pred_score_2, ref_score_2, raw_2 = self._judge_pair(prompt_b)
            raw_responses.append(raw_2)
            if pred_score_2 is not None and ref_score_2 is not None:
                scores_ref.append(ref_score_2)
                scores_pred.append(pred_score_2)

        if scores_pred and scores_ref:
            mean_pred = sum(scores_pred) / len(scores_pred)
            mean_ref = sum(scores_ref) / len(scores_ref)
            win = 1.0 if mean_pred > mean_ref else 0.0
            score.value = {
                'gpt_score': mean_pred,
                'win_rate': win,
            }
            score.main_score_name = 'gpt_score'
            score.metadata = {
                'reference_score': mean_ref,
                'pred_scores_per_pass': scores_pred,
                'reference_scores_per_pass': scores_ref,
                'judge_strategy': self.judge_strategy,
                'judge_model': getattr(self.llm_judge, 'model_id', 'unknown'),
                'do_swap': do_swap,
            }
            score.explanation = ' || '.join(raw_responses)
        else:
            logger.warning(f'AIR-Bench Chat: failed to parse judge response(s): {raw_responses!r}')
            score.value = {'gpt_score': 0.0, 'win_rate': 0.0}
            score.main_score_name = 'gpt_score'
            score.metadata = {'parse_failed': True, 'judge_raw': raw_responses}
        return score

    def _judge_pair(self, prompt: str) -> Tuple[Optional[float], Optional[float], str]:
        """Call the judge once and parse its two-integer response.

        Returns ``(score_assistant1, score_assistant2, raw_response)``. The two
        score positions correspond to the order embedded in ``prompt``.
        """
        try:
            judge = self.llm_judge
            if judge is None:
                return None, None, '<judge_error: LLM judge is not initialised>'
            raw = judge.judge(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        except Exception as e:
            logger.warning(f'AIR-Bench Chat: judge call failed: {e}')
            return None, None, f'<judge_error: {e}>'

        if raw is None:
            return None, None, ''

        nums = self._extract_judge_scores(raw)
        if len(nums) >= 2:
            try:
                a, b = float(nums[0]), float(nums[1])
                if 1 <= a <= 10 and 1 <= b <= 10:
                    return float(a), float(b), raw
            except ValueError:
                pass
        return None, None, raw

    @staticmethod
    def _extract_judge_scores(raw: str) -> List[str]:
        score_pattern = r'(?:10(?:\.0+)?|[0-9](?:\.\d+)?)'

        for line in raw.splitlines():
            cleaned = re.sub(r'(?i)assistant\s*[12]\s*(?:score)?', 'assistant', line)
            slash_nums = re.findall(rf'(?<![\w.])({score_pattern})\s*/\s*10(?![\w.])', cleaned)
            if len(slash_nums) == 2:
                return slash_nums
            nums = re.findall(rf'(?<![\w.])({score_pattern})(?![\w.])', cleaned)
            if len(nums) == 2:
                return nums

        cleaned = re.sub(r'(?i)assistant\s*[12]\s*(?:score)?', 'assistant', raw)
        slash_nums = re.findall(rf'(?<![\w.])({score_pattern})\s*/\s*10(?![\w.])', cleaned)
        if len(slash_nums) >= 2:
            return slash_nums[-2:]
        nums = re.findall(rf'(?<![\w.])({score_pattern})(?![\w.])', cleaned)
        return nums[-2:] if len(nums) >= 2 else nums
