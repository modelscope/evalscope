# flake8: noqa: E501
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import (
    DatasetDict,
    Sample,
    build_dataset_dict_from_record_map,
    resolve_snapshot_or_local_path,
)
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from .utils import (
    ANSWER_JUDGE_PROMPT,
    CHINESE_PROMPT,
    ENGLISH_PROMPT,
    STEP_JUDGE_PROMPT,
    criterion_points,
    extract_boxed_answers,
    is_chinese_exam,
    normalize_marking,
    parse_judge_correct,
    parse_judge_points,
    strip_boxed,
)

logger = get_logger()

# One subset per Olympiad exam paper, keyed by the record ``source`` field.
SUBSET_LIST = [
    'APhO_2025',
    'CPhO_2025',
    'EuPhO_2024',
    'EuPhO_2025',
    'F=MA_2024',
    'F=MA_2025',
    'IPhO_2024',
    'IPhO_2025',
    'NBPhO_2024',
    'NBPhO_2025',
    'PanMechanics_2024',
    'PanMechanics_2025',
    'PanPhO_2024',
    'PanPhO_2025',
]

DESCRIPTION = """
## Overview

HiPhO is the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. It compiles
13 recent Olympiad exams (2024-2025) spanning international and regional competitions, with mixed modalities that
range from text-only problems to diagram-based problems.

## Task Description

- **Task Type**: Free-form physics problem solving graded against official marking schemes
- **Input**: A physics problem (constants sheet + context + question), optionally with figures
- **Output**: A step-by-step solution ending with boxed final answers inside `<answer>...</answer>`
- **Modalities**: Text-only and text+figure (illustration / variable / data figures)

## Key Features

- 403 problems across 14 exam papers (IPhO, APhO, EuPhO, NBPhO, PanPhO, PanMechanics, CPhO, F=MA), each exam
  exposed as its own subset.
- English prompts are used for English exams and Chinese prompts for the Chinese exams (CPhO, PanMechanics),
  following the official language mapping.
- Two grading regimes reproduced from the paper, dispatched per problem:
  - **Step-level** for problems shipping an official marking scheme: the LLM judge scores every marking criterion
    and the awarded points are summed.
  - **Answer-level** for problems without a marking scheme: boxed final answers are matched against the ground
    truth by a rule-based math check, with an LLM judge as fallback.

## Evaluation Notes

- Requires an LLM judge: run with `judge_strategy='llm'` (or `'auto'`, which enables the judge for this benchmark)
  and provide `judge_model_args`. `judge_strategy='rule'` is not supported.
- Primary metric: `accuracy`, the per-problem awarded/attainable point ratio in `[0, 1]`, aggregated by mean per subset.
  For step-level problems the attainable maximum is the sum of the marking criteria; for problems with several
  official schemes (EuPhO, NBPhO) the highest-scoring scheme is used, matching the paper.
- This reports the normalized exam score per exam. It does not compute the paper's gold/silver/bronze medal
  thresholds, which require the raw point totals and official cutoffs.
- Solutions can be long and figure problems need vision input; give the evaluated model a generous
  `generation_config.max_tokens`. A solution truncated before its `<answer>` block yields no boxed answer and
  scores near zero for reasons unrelated to physics ability.
- Figures are sent inline as base64 and the largest is ~1.5 MB; set `max_image_bytes` in `dataset_args` if the
  served model enforces a smaller per-image limit.
- Resources: [Paper](https://arxiv.org/abs/2509.07894) | [GitHub](https://github.com/SciYu/HiPhO) |
  [Leaderboard](https://phyarena.github.io/)
"""


@register_benchmark(
    BenchmarkMeta(
        name='hipho',
        pretty_name='HiPhO',
        dataset_id='evalscope/HiPhO',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.MATH, Tags.QA],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2509.07894',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
    )
)
class HiPhOAdapter(VisionLanguageAdapter):
    """High school physics Olympiad benchmark graded by an LLM judge.

    Problems with an official marking scheme are graded step-by-step; problems
    without one are graded by matching their boxed final answers.
    """

    llm_judge_default = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Set during load(); used by record_to_sample to resolve figure paths.
        self.data_root: Optional[str] = None

    def load(self) -> Tuple[DatasetDict, None]:
        """Read the per-exam JSON files and their figures from the snapshot."""
        snapshot_dir = resolve_snapshot_or_local_path(self)
        self.data_root = os.path.join(snapshot_dir, 'data')

        record_map: Dict[str, List[Dict[str, Any]]] = {}
        for source in self.subset_list:
            json_path = os.path.join(self.data_root, f'{source}.json')
            if not os.path.exists(json_path):
                logger.warning(f'HiPhO exam file not found, skipping subset {source}: {json_path}')
                continue
            with open(json_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)

            # The optional leading {"information": ...} entry is the exam-wide
            # constants sheet shared by every problem in the paper.
            information = ''
            records: List[Dict[str, Any]] = []
            for entry in entries:
                if 'question' not in entry:
                    information = entry.get('information', information)
                    continue
                entry['information'] = information
                records.append(entry)
            record_map[source] = records

        return build_dataset_dict_from_record_map(
            record_map=record_map,
            sample_fields=self.record_to_sample,
            location=self.dataset_id,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
        ), None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Build a multimodal Sample from one physics problem."""
        source = record['source']
        template = CHINESE_PROMPT if is_chinese_exam(source) else ENGLISH_PROMPT
        prompt = template.format(
            information=record.get('information', ''),
            context=record.get('context', ''),
            question=record['question'],
        )

        content: List[Content] = [ContentText(text=prompt)]
        for image_ref in record.get('image_question') or []:
            image_b64 = self._load_figure(image_ref)
            if image_b64 is not None:
                content.append(ContentImage(image=image_b64))

        marking = normalize_marking(record.get('marking'))
        return Sample(
            input=[ChatMessageUser(content=content)],
            target='',  # ground truth lives in metadata; scoring is judge-based
            subset_key=source,
            metadata={
                'id': record['id'],
                'source': source,
                'question': record['question'],
                'answers': record.get('answer') or [],
                'marking': marking,
            },
        )

    def _load_figure(self, image_ref: str) -> Optional[str]:
        """Read a figure referenced by a problem and return a base64 data URI.

        ``image_ref`` comes from the dataset record, so it is confined to the
        snapshot directory: a record must not be able to read arbitrary files off
        disk. The reference itself is validated rather than its resolved target,
        because the HuggingFace hub cache legitimately symlinks snapshot files to a
        sibling ``blobs`` directory outside the snapshot.
        """
        normalized_ref = os.path.normpath(image_ref)
        if os.path.isabs(normalized_ref) or normalized_ref.split(os.sep)[0] == os.pardir:
            raise ValueError(f'HiPhO figure path escapes the dataset directory: {image_ref}')
        path = os.path.join(self.data_root, normalized_ref)
        if not os.path.exists(path):
            logger.warning(f'HiPhO figure not found: {path}')
            return None
        ext = os.path.splitext(path)[1].lower().lstrip('.') or 'png'
        with open(path, 'rb') as f:
            return self._image_bytes_to_base64(f.read(), default_format=ext)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        raise ValueError(
            'HiPhO is graded against official marking schemes and requires an LLM judge. '
            "Set judge_strategy='llm' (or 'auto') and provide judge_model_args."
        )

    def llm_match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        metadata = task_state.metadata or {}
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)

        if metadata.get('marking'):
            acc, detail = self._score_step_level(original_prediction, metadata)
        else:
            acc, detail = self._score_answer_level(original_prediction, metadata)

        score.value = {'acc': acc}
        score.main_score_name = 'acc'
        score.explanation = detail
        score.metadata = {
            'source': 'hipho_llm_judge',
            'grading': 'step_level' if metadata.get('marking') else 'answer_level',
            'judge_model': self.llm_judge.model_id,
        }
        # Surface the official answer in the review Gold column, which is otherwise
        # empty because grading is judge-based rather than reference-based.
        task_state.target = self._format_target(metadata)
        return score

    @staticmethod
    def _format_target(metadata: Dict[str, Any]) -> str:
        """Render the official answer for display in review files and the dashboard."""
        answers = [strip_boxed(a) for a in metadata['answers'] if (a or '').strip()]
        if answers:
            return ' | '.join(answers)
        # Open-ended problems ship no reference answer, only a marking scheme.
        criteria = sum(len(scheme) for scheme in metadata['marking'])
        return f'graded on {criteria} marking criteria'

    def _score_step_level(self, prediction: str, metadata: Dict[str, Any]) -> Tuple[float, str]:
        """Grade every criterion of each official scheme and keep the best scheme."""
        question = metadata['question']
        best_ratio = 0.0
        best_detail = ''
        for scheme_idx, scheme in enumerate(metadata['marking']):
            awarded = 0.0
            attainable = 0.0
            lines: List[str] = []
            for criterion in scheme:
                max_points = criterion_points(criterion)
                attainable += max_points
                prompt = STEP_JUDGE_PROMPT.format(question=question, prediction=prediction, criterion=criterion)
                response = self.llm_judge.judge(prompt)
                points = parse_judge_points(response, max_points)
                awarded += points
                lines.append(f'{points:g}/{max_points:g}')
            if attainable <= 0:
                # No criterion stated a point allocation, so the scheme cannot be
                # graded. Warn instead of silently reporting a zero score.
                logger.warning(
                    f'HiPhO sample {metadata["id"]} scheme {scheme_idx} has no parseable point '
                    f'allocation in its marking criteria; scoring it as 0.'
                )
                continue
            ratio = awarded / attainable
            if ratio >= best_ratio:
                best_ratio = ratio
                best_detail = f'scheme {scheme_idx}: {awarded:g}/{attainable:g} [' + ', '.join(lines) + ']'
        return best_ratio, best_detail

    def _score_answer_level(self, prediction: str, metadata: Dict[str, Any]) -> Tuple[float, str]:
        """Match each boxed final answer against the ground truth (rule first, judge fallback)."""
        from evalscope.metrics.math import math_equal

        gold_answers = [strip_boxed(a) for a in metadata['answers']]
        if not gold_answers:
            return 0.0, 'no ground-truth answer'

        pred_boxed = extract_boxed_answers(prediction)
        # The final answers correspond to the trailing boxed expressions, aligned
        # in order with the ground-truth sub-answers.
        aligned = pred_boxed[-len(gold_answers):] if pred_boxed else []

        correct = 0
        lines: List[str] = []
        for idx, gold in enumerate(gold_answers):
            pred = aligned[idx] if idx < len(aligned) else ''
            hit = bool(pred) and (math_equal(pred, gold) or self._judge_answer(metadata, pred, gold))
            correct += int(hit)
            lines.append(f'{"✓" if hit else "✗"}({pred or "∅"}|{gold})')
        return correct / len(gold_answers), ' '.join(lines)

    def _judge_answer(self, metadata: Dict[str, Any], prediction: str, gold: str) -> bool:
        """Fall back to the LLM judge for answer equivalence when rules do not match."""
        prompt = ANSWER_JUDGE_PROMPT.format(question=metadata['question'], given_answer=prediction, ground_truth=gold)
        return parse_judge_correct(self.llm_judge.judge(prompt))
