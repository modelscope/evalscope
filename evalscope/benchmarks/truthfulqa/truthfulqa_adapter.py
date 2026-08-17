# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

TruthfulQA is a benchmark designed to measure whether a language model generates truthful answers to questions. It specifically targets questions where humans might answer incorrectly due to common misconceptions, superstitions, or widely held but false beliefs.

## Task Description

- **Task Type**: Multiple-Choice Question Answering (Truthfulness Evaluation)
- **Input**: Question with variable number of answer choices (typically 4-8)
- **Output**: Correct answer letter
- **Categories**: 38 categories including Misconceptions, Superstitions, Conspiracies, Finance, Law, Health, etc.

## Key Features

- 817 questions spanning 38 diverse categories
- Specifically designed to expose model tendencies toward imitative falsehoods
- Questions where the best answer contradicts popular misconceptions
- Variable number of choices per question (MC1: single correct, MC2: multiple correct)
- Used as a core benchmark in the HuggingFace Open LLM Leaderboard

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on the `mc1` (single-correct) subset
- The `mc2` subset (multiple-correct) is also available for multi-label evaluation
- Only the `validation` split is available (no train split)
- Accuracy measures whether the model selects the truthful answer over plausible-sounding falsehoods
"""


@register_benchmark(
    BenchmarkMeta(
        name="truthfulqa_mc1",
        pretty_name="TruthfulQA MC1",
        tags=[Tags.HALLUCINATION, Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id="truthfulqa/truthful_qa",
        paper_url="https://arxiv.org/abs/2109.07958",
        metric_list=["acc"],
        subset_list=["multiple_choice"],
        few_shot_num=0,
        train_split=None,
        eval_split="validation",
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class TruthfulQAMC1Adapter(MultiChoiceAdapter):
    """
    TruthfulQA MC1 (single-correct) adapter.

    Each question has one correct answer among multiple distractors.
    The correct answer is the one with label=1 in mc1_targets.
    """

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        mc1 = record["mc1_targets"]
        choices = mc1["choices"]
        labels = mc1["labels"]
        # Find the index of the correct answer (label == 1)
        correct_idx = labels.index(1)
        # Convert index to letter (0 -> A, 1 -> B, ...)
        target = chr(ord("A") + correct_idx)
        return Sample(
            input=record["question"],
            choices=choices,
            target=target,
        )


@register_benchmark(
    BenchmarkMeta(
        name="truthfulqa_mc2",
        pretty_name="TruthfulQA MC2",
        tags=[Tags.HALLUCINATION, Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id="truthfulqa/truthful_qa",
        paper_url="https://arxiv.org/abs/2109.07958",
        metric_list=["acc"],
        subset_list=["multiple_choice"],
        few_shot_num=0,
        train_split=None,
        eval_split="validation",
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class TruthfulQAMC2Adapter(MultiChoiceAdapter):
    """
    TruthfulQA MC2 (multi-correct) adapter.

    Each question may have multiple correct answers among the choices.
    The correct answers are those with label=1 in mc2_targets.
    For evaluation, the model only needs to select any one of the correct answers.
    """

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        mc2 = record["mc2_targets"]
        choices = mc2["choices"]
        labels = mc2["labels"]
        # Find the first correct answer (label == 1) as the primary target
        correct_idx = labels.index(1)
        target = chr(ord("A") + correct_idx)
        return Sample(
            input=record["question"],
            choices=choices,
            target=target,
        )
