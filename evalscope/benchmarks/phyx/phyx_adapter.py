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
from .utils import (
    OPTION_LABELS,
    build_mc_judge_prompt,
    build_mc_question,
    build_oe_judge_prompt,
    build_oe_question,
    extract_mc_answer,
    extract_oe_answer,
    match_mc_answer,
    match_oe_answer,
    parse_judge_verdict,
    parse_options,
)

# The benchmark ships as one JSONL of problems plus a flat directory of figures.
DATA_DIR = 'test'
RECORD_FILE = 'PhyX_test.jsonl'
IMAGE_DIR = 'test_image'

# ``test`` is the full 3,000-problem set; the official ``testmini`` is its first 1,000 problems.
FULL_SPLIT = 'test'
MINI_SPLIT = 'test_mini'
MINI_SIZE = 1000

# One subset per core physics domain, keyed by the record ``category`` field. The subset names are
# file-safe identifiers because they become part of the prediction/review file names, which rules
# out the dataset's own 'Waves/Acoustics' spelling.
CATEGORY_TO_SUBSET = {
    'Mechanics': 'mechanics',
    'Electromagnetism': 'electromagnetism',
    'Thermodynamics': 'thermodynamics',
    'Waves/Acoustics': 'waves_acoustics',
    'Optics': 'optics',
    'Modern Physics': 'modern_physics',
}
SUBSET_LIST = list(CATEGORY_TO_SUBSET.values())

_SHARED_FEATURES = """
- 3,000 university-level problems (`test`) over 6 core domains and 25 sub-domains, each domain exposed
  as its own subset; `eval_split='test_mini'` selects the official 1,000-problem testmini set.
- Every problem is grounded in a figure that carries information the text does not restate, so the
  model must combine visual cues with implicit physical laws.
- 6 reasoning types are represented (physical model grounding, multi-formula, spatial relation,
  numerical, predictive and implicit condition reasoning).
- Uses the default *Text-DeRedundancy* input style of the paper: the simplified problem description
  plus the question, with the figure attached.
"""


class PhyXAdapter(VisionLanguageAdapter):
    """Shared loading for PhyX.

    The dataset is distributed as a JSONL of problems next to a directory of figures, so the records
    are read from the snapshot instead of a tabular split. Subclasses decide how a problem is asked
    (multiple-choice or open-ended) and how a reply is scored.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Set during load(); record_to_sample resolves figures relative to it.
        self.image_root: Optional[str] = None

    def load(self) -> Tuple[DatasetDict, None]:
        """Read the problem records and their figures from the snapshot."""
        if self.eval_split not in (FULL_SPLIT, MINI_SPLIT):
            raise ValueError(
                f"Unknown PhyX eval_split '{self.eval_split}'. Use '{FULL_SPLIT}' for the full "
                f"3,000-problem set or '{MINI_SPLIT}' for the official testmini subset."
            )
        unknown = [subset for subset in self.subset_list if subset not in SUBSET_LIST]
        if unknown:
            raise ValueError(f'Unknown PhyX subsets {unknown}. Valid subsets are: {SUBSET_LIST}')

        # Only fetch the problem file and its figures; the repository also holds ~1 GB of
        # pre-rendered parquet/TSV exports that this adapter does not use.
        snapshot_dir = resolve_snapshot_or_local_path(
            self, allow_file_pattern=[f'{DATA_DIR}/{RECORD_FILE}', f'{DATA_DIR}/{IMAGE_DIR}/*']
        )
        data_root = os.path.join(snapshot_dir, DATA_DIR)
        self.image_root = os.path.join(data_root, IMAGE_DIR)

        record_map: Dict[str, List[Dict[str, Any]]] = {subset: [] for subset in self.subset_list}
        with open(os.path.join(data_root, RECORD_FILE), 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                # The official testmini is the first 1,000 problems of the test set.
                if self.eval_split == MINI_SPLIT and int(record['index']) >= MINI_SIZE:
                    continue
                subset = CATEGORY_TO_SUBSET.get(record['category'])
                if subset is None:
                    raise ValueError(f'PhyX problem {record["index"]} has unknown category {record["category"]!r}.')
                if subset in record_map:
                    record_map[subset].append(record)

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
        options = parse_options(record['options'])
        if set(options) != set(OPTION_LABELS) or record['answer'] not in options:
            # Every PhyX problem offers exactly A-D, so a short or relabelled parse means the option
            # string was only partially read. Dropping the problem would leave the run reporting
            # fewer problems per domain than PhyX defines, and continuing would put an incomplete
            # question in front of the model.
            raise ValueError(
                f'PhyX problem {record["index"]}: parsed option labels {sorted(options)} with answer '
                f'{record["answer"]!r}, expected exactly {list(OPTION_LABELS)}.'
            )

        content: List[Content] = [
            ContentImage(image=self._load_figure(record['image'])),
            ContentText(text=self.build_question(record, options)),
        ]
        return Sample(
            input=[ChatMessageUser(content=content)],
            target=self.build_target(record, options),
            metadata={
                'index': record['index'],
                'category': record['category'],
                'subfield': record['subfield'],
                'reasoning_type': record['reasoning_type'],
            },
        )

    def _load_figure(self, image_name: str) -> str:
        """Read a figure from the snapshot and return it as a base64 data URI.

        The name comes from the dataset record, so it stays confined to the figure directory. Some
        figures carry JPEG data under a .png name (49 of the 3,000 released ones), so the MIME type
        is sniffed from the bytes rather than taken from the extension.
        """
        normalized = os.path.normpath(image_name)
        if os.path.isabs(normalized) or normalized.split(os.sep)[0] == os.pardir:
            raise ValueError(f'PhyX figure path escapes the dataset directory: {image_name}')
        with open(os.path.join(self.image_root, normalized), 'rb') as f:
            return self._image_bytes_to_base64(f.read(), default_format='png', guess_mimetype=True)

    def build_question(self, record: Dict[str, Any], options: Dict[str, str]) -> str:
        """Render the problem statement in the format this benchmark asks for."""
        raise NotImplementedError

    def build_target(self, record: Dict[str, Any], options: Dict[str, str]) -> str:
        """Return the ground-truth answer in the format this benchmark scores against."""
        raise NotImplementedError

    def _build_score(self, prediction: str, extracted: str, correct: bool, explanation: str) -> Score:
        score = Score(prediction=prediction, extracted_prediction=extracted)
        score.value = {'acc': float(correct)}
        score.main_score_name = 'acc'
        score.explanation = explanation
        return score


@register_benchmark(
    BenchmarkMeta(
        name='phyx_mc',
        pretty_name='PhyX-MC',
        dataset_id='evalscope/PhyX',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description=f"""
## Overview

PhyX is the first large-scale benchmark for physical reasoning in realistic, visually grounded
scenarios. This is its multiple-choice variant: each university-level physics problem is presented
with a figure and four answer options, and the model has to name the correct option letter.

## Task Description

- **Task Type**: Visual multiple-choice physics problem solving
- **Input**: A figure plus the problem description, question and four labelled options
- **Output**: A single option letter (A, B, C or D)
- **Domain**: University-level physics (mechanics, electromagnetism, thermodynamics, wave/acoustics,
  optics, modern physics)

## Key Features
{_SHARED_FEATURES}
- The official prompt is reproduced verbatim, including its instruction to answer with the option
  letter only, so scores stay comparable with the published numbers.

## Evaluation Notes

- Primary metric: `acc`, mean over problems, reported overall and per domain.
- Default scoring is the official string-level match: the chosen letter is extracted from the reply
  and compared with the ground truth, accepting replies that mark the correct option the way the
  prompt prints it (`D:`) or emphasises it (`**D**`).
- Setting `judge_strategy='llm'` (with `judge_model_args`) reproduces the official LLM-judged mode.
  The judge is only consulted for replies whose option letter could not be extracted, matching
  upstream.
- Figures are sent inline as base64 and the largest is ~5 MB; set `max_image_bytes` in `dataset_args`
  if the served model enforces a smaller per-image limit.
- Resources: [Paper](https://arxiv.org/abs/2505.15929) | [GitHub](https://github.com/NastyMarcus/PhyX)
  | [Project page](https://killthefullmoon.github.io/projects/PhyX/index.html)
""",
        paper_url='https://arxiv.org/abs/2505.15929',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split=FULL_SPLIT,
    )
)
class PhyXMCAdapter(PhyXAdapter):
    """PhyX in multiple-choice mode, scored by matching the chosen option letter."""

    def build_question(self, record: Dict[str, Any], options: Dict[str, str]) -> str:
        return build_mc_question(record['question_simply'], record['question'], options)

    def build_target(self, record: Dict[str, Any], options: Dict[str, str]) -> str:
        return record['answer']

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return extract_mc_answer(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        correct = match_mc_answer(filtered_prediction, original_prediction, reference)
        return self._build_score(original_prediction, filtered_prediction, correct, 'string match')

    def llm_match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score with the official judge, which only arbitrates replies without a clear letter.

        The pre-check is plain equality rather than the lenient ``match_mc_answer`` used in rule
        mode: upstream keeps the ``D:`` / ``**D**`` fallbacks out of its judged path, and accepting
        them here would credit a reply that committed to one letter while merely quoting the
        correct option's text.
        """
        if reference.strip().lower() == filtered_prediction.strip().lower():
            return self._build_score(original_prediction, filtered_prediction, True, 'string match')
        if filtered_prediction.strip() in OPTION_LABELS:
            # The reply committed to a different option; there is nothing for the judge to weigh.
            return self._build_score(original_prediction, filtered_prediction, False, 'string match')

        judge_response = self.llm_judge.judge(build_mc_judge_prompt(filtered_prediction, reference))
        correct = parse_judge_verdict(judge_response)
        return self._build_score(original_prediction, filtered_prediction, correct, f'LLM judge: {judge_response}')


@register_benchmark(
    BenchmarkMeta(
        name='phyx_oe',
        pretty_name='PhyX-OE',
        dataset_id='evalscope/PhyX',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.QA],
        description=f"""
## Overview

PhyX is the first large-scale benchmark for physical reasoning in realistic, visually grounded
scenarios. This is its open-ended variant: no options are shown, so the model has to derive the
answer of a university-level physics problem from the figure and state it.

## Task Description

- **Task Type**: Visual open-ended physics problem solving
- **Input**: A figure plus the problem description and question
- **Output**: A step-by-step derivation ending in the final answer (value with unit or a formula)
- **Domain**: University-level physics (mechanics, electromagnetism, thermodynamics, wave/acoustics,
  optics, modern physics)

## Key Features
{_SHARED_FEATURES}
- The official prompt is reproduced verbatim, including its request for step-by-step reasoning, so
  scores stay comparable with the published numbers.

## Evaluation Notes

- Primary metric: `acc`, mean over problems, reported overall and per domain.
- The final answer is read from `\\boxed{{...}}`, else from a 'final answer:' / 'correct answer:'
  statement, else the whole reply is compared. A reply truncated before its answer therefore scores
  0 for reasons unrelated to physics ability; give the model a generous `generation_config.max_tokens`.
- Answers are free-form values with units, so an LLM judge is used by default (the official
  recommendation): run with `judge_strategy='auto'` or `'llm'` and provide `judge_model_args`. The
  judge is only consulted when the answer does not already match as a string.
- `judge_strategy='rule'` falls back to the official string-level mode, which understates accuracy
  because equivalent spellings (`0.5 m` vs `50 cm`) do not match literally.
- Figures are sent inline as base64 and the largest is ~5 MB; set `max_image_bytes` in `dataset_args`
  if the served model enforces a smaller per-image limit.
- Resources: [Paper](https://arxiv.org/abs/2505.15929) | [GitHub](https://github.com/NastyMarcus/PhyX)
  | [Project page](https://killthefullmoon.github.io/projects/PhyX/index.html)
""",
        paper_url='https://arxiv.org/abs/2505.15929',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split=FULL_SPLIT,
    )
)
class PhyXOEAdapter(PhyXAdapter):
    """PhyX in open-ended mode, scored by comparing the final answer with the ground truth."""

    llm_judge_default = True

    def build_question(self, record: Dict[str, Any], options: Dict[str, str]) -> str:
        return build_oe_question(record['question_simply'], record['question'])

    def build_target(self, record: Dict[str, Any], options: Dict[str, str]) -> str:
        # The open-ended ground truth is the text of the correct option.
        return options[record['answer']]

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return extract_oe_answer(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        correct = match_oe_answer(filtered_prediction, original_prediction, reference)
        return self._build_score(original_prediction, filtered_prediction, correct, 'string match')

    def llm_match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Score with the official judge, which decides whether the answers are equivalent."""
        if reference.strip().lower() == filtered_prediction.strip().lower():
            return self._build_score(original_prediction, filtered_prediction, True, 'string match')

        judge_response = self.llm_judge.judge(build_oe_judge_prompt(filtered_prediction, reference))
        correct = parse_judge_verdict(judge_response)
        return self._build_score(original_prediction, filtered_prediction, correct, f'LLM judge: {judge_response}')
