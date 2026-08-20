# flake8: noqa: E501
import json
import os
from pydantic import BaseModel, Field, create_model
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import (
    DatasetDict,
    Sample,
    build_dataset_dict_from_record_map,
    resolve_snapshot_or_local_path,
)
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeDefinition, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
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
    strip_boxed,
)

logger = get_logger()


class AnswerVerdict(BaseModel):
    """The judge's [Correct]/[Incorrect] answer-level verdict, as JSON."""
    correct: bool


ANSWER_CONTRACT = OutputContract(schema_model=AnswerVerdict)


def _step_grade_model(max_points: float):
    """Per-criterion schema whose upper bound matches the criterion's stated allocation."""
    return create_model(
        'StepGrade',
        awarded=(float, Field(ge=0.0, le=float(max_points))),
    )


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

- Requires an LLM judge: set `judge.strategy='llm'` (or `'auto'`, which enables the judge for this benchmark)
  and provide `judge.models`. `judge.strategy='rule'` is not supported.
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

    scoring_policy = ScoringPolicy.JUDGE_ONLY

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

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        immediate, skip_reason = self._rule_score(
            context.original_prediction, context.filtered_prediction, context.reference, context.task_state
        )
        if immediate is not None:
            assert skip_reason is not None
            return JudgeDefinition.skip(immediate, reason=skip_reason)
        return JudgeDefinition.workflow(
            cases=self._build_cases(context),
            request=self._build_request,
            reduce=self._reduce_verdicts,
            main_score_name='acc',
            finalize=self._finalize_score
        )

    def _rule_score(self, original_prediction: str, filtered_prediction: str, reference: str,
                    task_state: TaskState) -> Tuple[Optional[Score], Optional[str]]:
        # Both flows short-circuit before touching the judge when there is nothing to grade.
        metadata = task_state.metadata or {}
        if metadata.get('marking'):
            if not any(scheme for scheme in metadata['marking']):
                return (
                    self._empty_score(filtered_prediction, original_prediction, task_state, 'no marking criteria'),
                    'no_marking_criteria',
                )
        else:
            if not [strip_boxed(a) for a in metadata['answers']]:
                return (
                    self._empty_score(filtered_prediction, original_prediction, task_state, 'no ground-truth answer'),
                    'missing_ground_truth_answer',
                )
        task_state.target = self._format_target(metadata)
        return None, None

    def _empty_score(self, filtered: str, original: str, task_state: TaskState, reason: str) -> Score:
        task_state.target = self._format_target(task_state.metadata or {})
        return Score(
            extracted_prediction=filtered,
            prediction=original,
            value={'acc': 0.0},
            main_score_name='acc',
            explanation=reason,
        )

    def _build_cases(self, context: JudgeContext) -> List[JudgeCase]:
        metadata = context.task_state.metadata or {}
        if metadata.get('marking'):
            return self._build_step_cases(metadata)
        return self._build_answer_cases(metadata, context.original_prediction)

    def _build_step_cases(self, metadata: Dict[str, Any]) -> List[JudgeCase]:
        cases: List[JudgeCase] = []
        for scheme_idx, scheme in enumerate(metadata['marking']):
            for crit_idx, criterion in enumerate(scheme):
                max_points = criterion_points(criterion)
                if max_points <= 0:
                    # A criterion with no parseable point allocation cannot be judged; count it as
                    # zero, as the previous evaluator did, without spending a judge call.
                    continue
                cases.append(
                    JudgeCase(
                        case_id=f'step:{scheme_idx}:{crit_idx}',
                        output_contract=OutputContract(schema_model=_step_grade_model(max_points)),
                        metadata={
                            'kind': 'step',
                            'scheme_idx': scheme_idx,
                            'criterion': criterion,
                            'max_points': max_points,
                        },
                    )
                )
        return cases

    def _build_answer_cases(self, metadata: Dict[str, Any], prediction: str) -> List[JudgeCase]:
        from evalscope.metrics.math import math_equal

        gold_answers = [strip_boxed(a) for a in metadata['answers']]
        pred_boxed = extract_boxed_answers(prediction)
        aligned = pred_boxed[-len(gold_answers):] if pred_boxed else []
        cases: List[JudgeCase] = []
        for idx, gold in enumerate(gold_answers):
            pred = aligned[idx] if idx < len(aligned) else ''
            # Rule check first: only ambiguous cases go to the judge, mirroring the old flow.
            if pred and math_equal(pred, gold):
                continue
            if not pred:
                continue
            cases.append(
                JudgeCase(
                    case_id=f'answer:{idx}',
                    output_contract=ANSWER_CONTRACT,
                    metadata={
                        'kind': 'answer',
                        'answer_idx': idx,
                        'gold': gold,
                        'pred': pred,
                    },
                )
            )
        return cases

    def _build_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        metadata = context.task_state.metadata or {}
        if case.metadata['kind'] == 'step':
            prompt = STEP_JUDGE_PROMPT.format(
                question=metadata['question'],
                prediction=context.original_prediction,
                criterion=case.metadata['criterion'],
            )
        else:
            prompt = ANSWER_JUDGE_PROMPT.format(
                question=metadata['question'],
                given_answer=case.metadata['pred'],
                ground_truth=case.metadata['gold'],
            )
        prompt += case.output_contract.instruction()
        return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

    def _reduce_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        metadata = context.task_state.metadata or {}
        if metadata.get('marking'):
            return self._reduce_step(case_verdicts, metadata)
        return self._reduce_answer(case_verdicts, metadata, context.original_prediction)

    def _reduce_step(self, case_verdicts, metadata: Dict[str, Any]) -> ReducedVerdict:
        verdicts_by_case = {verdict.case_id: verdict for verdict in case_verdicts}
        best_ratio = 0.0
        best_detail = ''
        for scheme_idx, scheme in enumerate(metadata['marking']):
            awarded = 0.0
            attainable = 0.0
            lines: List[str] = []
            for crit_idx, criterion in enumerate(scheme):
                max_points = criterion_points(criterion)
                if max_points <= 0:
                    continue
                attainable += max_points
                verdict = verdicts_by_case.get(f'step:{scheme_idx}:{crit_idx}')
                # `Field(le=max_points)` bounds the judge's number to the criterion's allocation.
                points = float(verdict.value.awarded) if verdict is not None else 0.0
                awarded += points
                lines.append(f'{points:g}/{max_points:g}')
            if attainable <= 0:
                continue
            ratio = awarded / attainable
            if ratio >= best_ratio:
                best_ratio = ratio
                best_detail = f'scheme {scheme_idx}: {awarded:g}/{attainable:g} [' + ', '.join(lines) + ']'
        return ReducedVerdict(
            value={'acc': best_ratio},
            metadata={
                'grading': 'step_level',
                'detail': best_detail
            },
        )

    def _reduce_answer(self, case_verdicts, metadata: Dict[str, Any], prediction: str) -> ReducedVerdict:
        from evalscope.metrics.math import math_equal

        gold_answers = [strip_boxed(a) for a in metadata['answers']]
        pred_boxed = extract_boxed_answers(prediction)
        aligned = pred_boxed[-len(gold_answers):] if pred_boxed else []
        verdicts_by_case = {verdict.case_id: verdict for verdict in case_verdicts}
        correct = 0
        lines: List[str] = []
        for idx, gold in enumerate(gold_answers):
            pred = aligned[idx] if idx < len(aligned) else ''
            if pred and math_equal(pred, gold):
                hit = True
            else:
                verdict = verdicts_by_case.get(f'answer:{idx}')
                hit = bool(verdict) and verdict.value.correct
            correct += int(hit)
            lines.append(f'{"✓" if hit else "✗"}({pred or "∅"}|{gold})')
        return ReducedVerdict(
            value={'acc': correct / len(gold_answers) if gold_answers else 0.0},
            metadata={
                'grading': 'answer_level',
                'detail': ' '.join(lines)
            },
        )

    def _finalize_score(self, score: Score, review, context) -> Score:
        score.explanation = review.metadata.get('detail', '')
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
