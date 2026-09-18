from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, Field

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    OutputContract,
    Placement,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from evalscope.api.model import GenerateConfig, Model
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags

SUBSETS = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']
REFERENCE_CATEGORIES = {'reasoning', 'math', 'coding'}
TEMPERATURE_CONFIG = {
    'writing': 0.7,
    'roleplay': 0.7,
    'extraction': 0.0,
    'math': 0.0,
    'coding': 0.0,
    'reasoning': 0.0,
    'stem': 0.1,
    'humanities': 0.1,
}
FALLBACK_REFERENCES = {
    34377376: [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Random Joke Generator</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
        button { font-size: 20px; padding: 10px 20px; cursor: pointer; }
    </style>
    <script>
        function showRandomJoke() {
            const jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "Why did the chicken go to the seance? To get to the other side.",
                "Why don't some couples go to the gym? Because some relationships don't work out.",
                "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them!"
            ];
            document.getElementById("jokeDisplay").innerHTML = jokes[Math.floor(Math.random() * jokes.length)];
        }
    </script>
</head>
<body>
    <h1>Random Joke Generator</h1>
    <button onclick="showRandomJoke()">Show me a joke!</button>
    <p id="jokeDisplay"></p>
</body>
</html>""",
        'Add `#jokeDisplay { color: red; }` to the CSS in the style block to display jokes in red.',
    ],
}

DESCRIPTION = """
## Overview

MT-Bench evaluates chat assistants on challenging, open-ended multi-turn conversations. It was introduced with the
LLM-as-a-judge study and uses a strong judge model to measure response quality beyond closed-form benchmarks.

## Task Description

- **Task Type**: Two-turn open-ended conversational generation with LLM-as-a-judge scoring
- **Input**: An initial user request followed by a user follow-up that depends on the first assistant response
- **Output**: Assistant responses to both turns, with the full conversation preserved for second-turn grading
- **Domain**: Writing, roleplay, extraction, reasoning, mathematics, coding, STEM, and humanities

## Key Features

- Contains 80 human-authored conversations, with 10 two-turn questions in each of eight categories.
- The adapter preserves the generated first answer in the second-turn context, matching the official conversation flow.
- The official protocol uses category temperatures and a maximum of 1,024 generated tokens when no task-level maximum
  is configured.
- The ModelScope mirror supplies the reference answers used for the official reference-guided reasoning, math, and
  coding grading prompts. Its one missing coding reference is supplied from the official FastChat evaluator.

## Evaluation Notes

- The official MT-Bench protocol uses `gpt-4` as the single-answer judge and reports the mean of all valid first- and
  second-turn ratings on a 1-10 scale.
- Configure `judge.models` to supply `gpt-4` for official-comparable runs. Compatible custom judge models are allowed,
  but their scores must not be presented as directly comparable to the official GPT-4 results.
- Reasoning, math, and coding prompts include the dataset reference answers; other categories use the general quality
  rubric covering helpfulness, relevance, accuracy, depth, creativity, and detail.
- Judge replies must satisfy EvalScope's JSON output contract. Parse and transport failures exclude the sample rather
  than assigning a zero score.
- Resources: [Paper](https://arxiv.org/abs/2306.05685) |
  [Official implementation](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) |
  [Dataset](https://modelscope.cn/datasets/HuggingFaceH4/mt_bench_prompts)
"""


class MTBenchVerdict(BaseModel):
    """One 1-10 MT-Bench judge rating."""

    explanation: str = ''
    score: int = Field(ge=1, le=10)


RATING_CONTRACT = OutputContract(schema_model=MTBenchVerdict)


@register_benchmark(
    BenchmarkMeta(
        name='mt_bench',
        pretty_name='MT-Bench',
        dataset_id='HuggingFaceH4/mt_bench_prompts',
        tags=[Tags.QA, Tags.INSTRUCTION_FOLLOWING, Tags.MULTI_TURN, Tags.REASONING, Tags.CODING, Tags.MATH],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2306.05685',
        subset_list=SUBSETS,
        default_subset='default',
        eval_split='train',
        train_split=None,
        few_shot_num=0,
        metric_list=['judge_score', 'first_turn_judge_score', 'second_turn_judge_score'],
        primary_metric='judge_score',
        aggregation='mean',
        judge_config={'strategy': 'llm', 'models': [{'model_id': 'gpt-4'}]},
        prompt_template=None,
        evaluation_version='v1.0',
    )
)
class MTBenchAdapter(DefaultDataAdapter):
    """Run MT-Bench's two generation turns and its official single-answer grading protocol."""

    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert one ModelScope MT-Bench prompt pair into an evaluation sample."""
        turns = record['prompt']
        category = record['category']
        if category not in SUBSETS:
            raise ValueError(f'Unknown MT-Bench category: {category}')
        if not isinstance(turns, list) or len(turns) != 2 or not all(isinstance(turn, str) for turn in turns):
            raise ValueError(f'MT-Bench prompt {record.get("prompt_id")} must contain exactly two string turns.')

        prompt_id = record['prompt_id']
        references = record.get('reference') or FALLBACK_REFERENCES.get(prompt_id, [])
        if category in REFERENCE_CATEGORIES and len(references) != 2:
            raise ValueError(f'MT-Bench {category} prompt {prompt_id} has no two-turn reference answer.')

        return Sample(
            input=[ChatMessageUser(content=turns[0])],
            target='',
            subset_key=category,
            metadata={
                'category': category,
                'prompt_id': prompt_id,
                'turns': turns,
                'references': references,
            },
        )

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        """Generate the two dependent assistant responses required by MT-Bench."""
        metadata = sample.metadata or {}
        category = metadata['category']
        config = self._generation_config(model, category)
        first_messages = list(sample.input)
        first_output = model.generate(input=first_messages, config=config)
        second_messages = [
            *first_messages,
            first_output.message,
            ChatMessageUser(content=metadata['turns'][1]),
        ]
        second_output = model.generate(input=second_messages, config=config)
        return InferenceResult(output=second_output, messages=[*second_messages, second_output.message])

    @staticmethod
    def _generation_config(model: Model, category: str) -> GenerateConfig:
        """Apply the official category temperature without adding benchmark-specific user options."""
        max_tokens = model.config.max_tokens if model.config.max_tokens is not None else 1024
        return model.config.model_copy(update={'temperature': TEMPERATURE_CONFIG[category], 'max_tokens': max_tokens})

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        """Declare the official first- and second-turn single-answer judge cases."""
        metadata = context.task_state.metadata or {}
        category = metadata['category']
        references = metadata.get('references') or []
        cases = [
            JudgeCase(
                case_id='first_turn',
                output_contract=RATING_CONTRACT,
                metadata={'turn_index': 0, 'category': category, 'reference': self._reference(references, 0, category)},
            ),
            JudgeCase(
                case_id='second_turn',
                output_contract=RATING_CONTRACT,
                metadata={'turn_index': 1, 'category': category, 'reference': self._reference(references, 1, category)},
            ),
        ]
        return JudgeDefinition.workflow(
            cases=cases,
            request=self._build_judge_request,
            reduce=self._reduce_verdicts,
            main_score_name='judge_score',
        )

    @staticmethod
    def _reference(references: List[str], turn_index: int, category: str) -> str:
        if category not in REFERENCE_CATEGORIES:
            return ''
        return references[turn_index]

    @staticmethod
    def _build_judge_request(
        case: JudgeCase,
        placement: Placement,
        completed_cases: Sequence[CaseVerdict],
        context: JudgeContext,
    ) -> JudgeRequest:
        del placement, completed_cases
        metadata = context.task_state.metadata or {}
        turns = metadata['turns']
        assistant_answers = [
            message.text for message in context.task_state.messages if isinstance(message, ChatMessageAssistant)
        ]
        if len(assistant_answers) != 2:
            raise ValueError('MT-Bench requires exactly two generated assistant messages.')

        turn_index = case.metadata['turn_index']
        if turn_index == 0:
            prompt, system_prompt = MTBenchAdapter._first_turn_prompt(
                question=turns[0],
                answer=assistant_answers[0],
                reference=case.metadata['reference'],
            )
        else:
            prompt, system_prompt = MTBenchAdapter._second_turn_prompt(
                turns=turns,
                answers=assistant_answers,
                references=metadata.get('references') or [],
                use_reference=case.metadata['category'] in REFERENCE_CATEGORIES,
            )
        return JudgeRequest(
            messages=[
                ChatMessageSystem(content=system_prompt),
                ChatMessageUser(content=prompt + case.output_contract.instruction()),
            ]
        )

    @staticmethod
    def _first_turn_prompt(question: str, answer: str, reference: str) -> tuple[str, str]:
        reference_section = ''
        instruction = (
            'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to '
            'the user question displayed below. Your evaluation should consider factors such as the helpfulness, '
            'relevance, accuracy, depth, creativity, and level of detail of the response.'
        )
        if reference:
            instruction = (
                'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant '
                'to the user question displayed below. Your evaluation should consider correctness and helpfulness. '
                "You will be given a reference answer and the assistant's answer. Begin by comparing the assistant's "
                'answer with the reference answer and identify any mistakes.'
            )
            reference_section = f'\n\n[The Start of Reference Answer]\n{reference}\n[The End of Reference Answer]'
        prompt = (
            f'[Instruction]\n{instruction}\n\n[Question]\n{question}{reference_section}\n\n'
            f"[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
        )
        return prompt, 'You are a helpful assistant.'

    @staticmethod
    def _second_turn_prompt(
        turns: List[str], answers: List[str], references: List[str], use_reference: bool
    ) -> tuple[str, str]:
        reference_section = ''
        system_prompt = (
            'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to '
            'the user question displayed below. Your evaluation should consider factors such as the helpfulness, '
            'relevance, accuracy, depth, creativity, and level of detail of the response. You should focus on the '
            "assistant's answer to the second user question. Begin your evaluation by providing a short explanation. "
            'Be as objective as possible.'
        )
        if use_reference:
            if len(references) != 2:
                raise ValueError('MT-Bench reference-guided judging requires two reference answers.')
            system_prompt = (
                'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant '
                'to the user question. Your evaluation should consider correctness and helpfulness. You will be given '
                "a reference answer and the assistant's answer. You should focus on the assistant's answer to the "
                "second question. Begin by comparing the assistant's answer with the reference answer and identify any mistakes."
            )
            reference_section = (
                '<|The Start of Reference Answer|>\n\n'
                f'### User:\n{turns[0]}\n\n### Reference answer:\n{references[0]}\n\n'
                f'### User:\n{turns[1]}\n\n### Reference answer:\n{references[1]}\n\n'
                '<|The End of Reference Answer|>\n\n\n'
            )
        prompt = (
            f"{reference_section}<|The Start of Assistant's Conversation with User|>\n\n"
            f'### User:\n{turns[0]}\n\n### Assistant:\n{answers[0]}\n\n'
            f'### User:\n{turns[1]}\n\n### Assistant:\n{answers[1]}\n\n'
            "<|The End of Assistant's Conversation with User|>"
        )
        return prompt, system_prompt

    @staticmethod
    def _reduce_verdicts(case_verdicts: Sequence[CaseVerdict], context: JudgeContext) -> ReducedVerdict:
        del context
        scores = {int(verdict.metadata['turn_index']): float(verdict.value.score) for verdict in case_verdicts}
        if set(scores) != {0, 1}:
            raise ValueError('MT-Bench requires valid judge verdicts for both turns.')
        first_turn_score = scores[0]
        second_turn_score = scores[1]
        return ReducedVerdict(
            value={
                'judge_score': (first_turn_score + second_turn_score) / 2,
                'first_turn_judge_score': first_turn_score,
                'second_turn_judge_score': second_turn_score,
            },
            metadata={
                'first_turn_score': first_turn_score,
                'second_turn_score': second_turn_score,
            },
        )
