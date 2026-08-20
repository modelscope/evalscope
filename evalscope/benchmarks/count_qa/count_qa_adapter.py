# flake8: noqa: E501
import re
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

DESCRIPTION = """
## Overview

CountQA probes object counting, a basic perceptual skill that multimodal models are largely
unevaluated on. Its images were hand-captured in everyday environments and deliberately feature
high object density, clutter and occlusion, so counting cannot be solved by detecting a handful of
well-separated objects.

## Task Description

- **Task Type**: Free-form Visual Question Answering (object counting)
- **Input**: A real-world photograph + a counting question (e.g. "How many jackets are there?")
- **Output**: A single integer
- **Domain**: Everyday scenes — groceries, kitchenware, tools, clothing, office and outdoor objects

## Key Features

- 1,528 question-answer pairs over 1,001 images; an image may carry several questions
- Ground-truth counts were annotated *in situ* during capture rather than post-hoc, and range from 0 to 400
- Questions include compositional ones that require summing over several object types
- Roughly half the images are cluttered rather than focused on a single subject (recorded as
  ``is_focused`` in each sample's metadata), and scene categories are recorded as ``categories``

## Evaluation Notes

- Default evaluation uses the **test** split as a single subset
- Primary metric: **Accuracy** (`accuracy`) — Exact Match against the ground-truth integer
- Secondary metric: **relaxed_acc** — the paper's Relaxed Accuracy, counting a prediction correct
  when it is within 5% of the ground truth
- The paper's system prompt is used as-is; it constrains the reply to a bare integer
- Answer parsing takes the reply if it is already an integer, otherwise its first integer — the
  rule the paper states for its rewriter LLM. A reply with no digit scores 0, so `max_tokens` must
  leave the model room to reach its answer; a model that narrates its count ("row 1 has 3 ...") is
  scored on the first number it mentions rather than on its stated total
- Scoring is deterministic arithmetic and needs no LLM judge: keep `judge.strategy` at `rule` or
  `auto`, since `llm` replaces both metrics with a generic judge score. To read a different number
  out of a model that ignores the output format, prepend a per-run filter such as
  `filters={'regex': {'regex_pattern': '(\\d+)', 'group_select': -1}}` (last number) via
  `dataset_args` rather than editing the adapter
- [Paper](https://arxiv.org/abs/2508.06585)
"""

# Verbatim system prompt from Appendix C of the paper.
SYSTEM_PROMPT = (
    'You are a helpful assistant that counts the number of items in an image. The user will provide an image '
    'and ask a question about the number of a certain type of item in the image. If the user question is '
    'referring to multiple objects, it means that you need to provide a sum of the number of items. You will '
    'count the number of items and return the number as an integer. Your output should STRICTLY be a single '
    'integer and nothing else.'
)

# Tolerance of the paper's Relaxed Accuracy metric.
RELAXED_TOLERANCE = 0.05

_INTEGER_PATTERN = re.compile(r'\d+')


def parse_count(prediction: str) -> Optional[int]:
    """Parse the predicted count from a model reply.

    The system prompt asks for a bare integer, so a compliant reply is used directly after
    stripping the punctuation and markdown emphasis models add around it. For a verbose reply the
    first integer is taken, which is the rule the paper states for its rewriter LLM. Note the limit
    of that rule: in a reply that counts out loud (``'Row 1 has 3, row 2 has 4. Total: 7'``) the
    first integer is a row label, not the answer. ``None`` means the reply contains no digit at all,
    which scores 0 rather than being replaced by a guess.

    Args:
        prediction (str): Raw model reply.

    Returns:
        Optional[int]: The predicted count, or ``None`` if the reply contains no digit.
    """
    cleaned = prediction.strip().strip('*`_"\'.,:; \n')
    if _INTEGER_PATTERN.fullmatch(cleaned):
        return int(cleaned)
    match = _INTEGER_PATTERN.search(prediction)
    return int(match.group()) if match else None


@register_benchmark(
    BenchmarkMeta(
        name='count_qa',
        pretty_name='CountQA',
        dataset_id='evalscope/CountQA',
        tags=[Tags.MULTI_MODAL, Tags.QA, Tags.REASONING],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2508.06585',
        metric_list=['acc', 'relaxed_acc'],
        primary_metric='accuracy',
        eval_split='test',
        system_prompt=SYSTEM_PROMPT,
    )
)
class CountQAAdapter(VisionLanguageAdapter):
    """Data adapter for evalscope/CountQA.

    Each record is one image holding aligned ``questions`` / ``answers`` lists, so
    ``record_to_sample`` returns one sample per question, all sharing the encoded image.
    Scoring is deterministic: the predicted integer is compared with the ground-truth count
    exactly (Exact Match) and within the paper's 5% tolerance (Relaxed Accuracy).
    """

    def record_to_sample(self, record: Dict[str, Any]) -> List[Sample]:
        """Expand one image record into one sample per counting question."""
        # Both the remote loader and the local parquet loader set decode=False on Image columns,
        # so the field always arrives as a {'bytes': ..., 'path': ...} dict.  The dataset mixes
        # JPEG and PNG images, hence the sniffed MIME type rather than a fixed one.
        image_base64 = self._image_bytes_to_base64(record['image']['bytes'], guess_mimetype=True)

        samples: List[Sample] = []
        for question, answer in zip(record['questions'], record['answers']):
            content_list: List[Content] = [
                ContentImage(image=image_base64),
                ContentText(text=question),
            ]
            samples.append(
                Sample(
                    input=[ChatMessageUser(content=content_list)],
                    target=str(answer).strip(),
                    metadata={
                        'is_focused': record.get('is_focused'),
                        'categories': record.get('categories'),
                    },
                )
            )
        return samples

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Return the predicted count as a string, or an empty string if the reply has none."""
        count = parse_count(prediction)
        return '' if count is None else str(count)

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score the predicted count with Exact Match and Relaxed Accuracy."""
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)

        reference_count = int(reference)
        if filtered_prediction:
            predicted_count = int(filtered_prediction)
            exact = float(predicted_count == reference_count)
            # A ground truth of 0 has no meaningful relative tolerance, so it stays exact.
            relaxed = exact if reference_count == 0 else float(
                abs(predicted_count - reference_count) <= RELAXED_TOLERANCE * reference_count
            )
        else:
            exact = relaxed = 0.0

        score.value = {'acc': exact, 'relaxed_acc': relaxed}
        score.main_score_name = 'acc'
        return score
