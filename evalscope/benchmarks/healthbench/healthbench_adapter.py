import copy
import os
from pydantic import BaseModel
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample, load_local_file_dataset, resolve_snapshot_or_local_path
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeDefinition, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages.chat_message import ChatMessageUser, dict_to_chat_message
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


class RubricGrade(BaseModel):
    """The judge's reply per rubric item; GRADER_TEMPLATE already asks for exactly these keys."""
    explanation: str = ''
    criteria_met: bool


RUBRIC_CONTRACT = OutputContract(schema_model=RubricGrade)

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()  # noqa: E501

# Available subsets in the HealthBench dataset
# Each subset focuses on different aspects of health-related conversations
SUBSET_LIST = [
    'emergency_referrals',  # Situations requiring immediate medical attention
    'communication',  # Communication skills and patient interaction
    'complex_responses',  # Complex medical scenarios requiring detailed responses
    'hedging',  # Appropriate uncertainty and hedging in medical advice
    'health_data_tasks',  # Tasks involving health data analysis
    'global_health',  # Global health perspectives and cultural considerations
    'context_seeking',  # Ability to seek additional context when needed
]

# Available versions of the dataset
VERSION = [
    'Consensus',
    'Hard',
    'All',
]

# Mapping of version names to their corresponding data files
VERSION_FILE = {
    'All': '2025-05-07-06-14-12_oss_eval.jsonl',  # Complete dataset
    'Consensus': 'consensus_2025-05-09-20-00-46.jsonl',  # Consensus subset
    'Hard': 'hard_2025-05-08-21-00-10.jsonl',  # Hard examples subset
}


@register_benchmark(
    BenchmarkMeta(
        name='health_bench',
        pretty_name='HealthBench',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.MEDICAL],
        description="""
## Overview

HealthBench is a comprehensive benchmark designed to measure AI capabilities for health-related tasks. Built in partnership with 262 physicians from 60 countries, it includes 5,000 realistic health conversations with custom physician-created rubrics.

## Task Description

- **Task Type**: Medical Conversation Evaluation
- **Input**: Health-related conversation with patient queries
- **Output**: Medical advice/response evaluated against physician rubrics
- **Domains**: Medical advice, health communication, clinical knowledge

## Key Features

- 5,000 realistic health conversations
- Custom rubrics created by 262 physicians
- Seven evaluation categories:
  - `emergency_referrals`: Urgent medical situations
  - `communication`: Patient interaction quality
  - `complex_responses`: Detailed medical scenarios
  - `hedging`: Appropriate uncertainty expression
  - `health_data_tasks`: Health data analysis
  - `global_health`: Cross-cultural considerations
  - `context_seeking`: Information gathering ability

## Evaluation Notes

- Default version: **Consensus** (also available: Hard, All)
- Multiple evaluation metrics:
  - Communication Quality, Instruction Following, Accuracy
  - Context Awareness, Completeness
- Uses LLM judge with physician-designed rubric items
- Aggregation: **clipped_mean** for robust scoring
""",  # noqa: E501
        dataset_id='openai-mirror/healthbench',
        subset_list=SUBSET_LIST,
        metric_list=[
            'communication_quality',
            'instruction_following',
            'accuracy',
            'context_awareness',
            'completeness',
        ],
        primary_metric='accuracy',
        aggregation='clipped_mean',
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='Answer the question:\n\n{question}',
        extra_params={
            'version': {
                'type': 'str',
                'description': f'Dataset file version, choices: {VERSION}.',
                'value': VERSION[0],
                'choices': VERSION
            }
        }
    )
)
class HealthBenchAdapter(DefaultDataAdapter):
    """
    Adapter for the HealthBench dataset that handles loading health conversation data
    and evaluating AI responses using physician-created rubrics.

    This adapter supports multiple dataset versions and uses LLM judges to evaluate
    responses against detailed medical criteria.
    """
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, *args, **kwargs):
        """
        Initialize the HealthBench adapter.

        Sets up default configuration including:
        - LLM judge evaluation
        - Dataset version selection
        - Subset reformatting
        """
        super().__init__(*args, **kwargs)

        self.reformat_subset = True
        # Get version from extra parameters, default to first version if not specified
        self.version = self.extra_params.get('version', VERSION[0])
        # Map version to corresponding data file
        self.version_file = VERSION_FILE[self.version]

    def load(self):
        """
        Load the HealthBench dataset from local or remote source.

        Returns:
            tuple: (test_dataset, None) where test_dataset is a DatasetDict
                   containing the loaded data split by subsets
        """
        dataset_path = resolve_snapshot_or_local_path(self, allow_file_pattern=self.version_file)
        dataset = load_local_file_dataset(
            adapter=self,
            dataset_path=dataset_path,
            subset=os.path.splitext(self.version_file)[0],  # NOTE: using hardcoded test subset
            split=self.eval_split,
            sample_fields=self.record_to_sample,
            limit=None,
            repeats=1,
            shuffle=self.shuffle,
        )

        # Convert to DatasetDict and apply subset filtering and limiting
        test_dataset = DatasetDict.from_dataset(
            dataset=dataset, subset_list=self.subset_list, limit=self.limit, repeats=self.repeats
        )

        return test_dataset, None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a raw data record to a Sample object.

        Args:
            record: Raw data record containing prompt, tags, and metadata

        Returns:
            Sample: Formatted sample with input messages, theme, and metadata
        """
        # Convert prompt messages to chat message objects
        input_messages = [dict_to_chat_message(message) for message in record['prompt']]
        # Extract theme from example tags, default to 'Unknown' if no tags
        tags = record['example_tags']
        theme = tags[0].split(':')[1].strip() if len(tags) > 0 else 'Unknown'
        return Sample(input=input_messages, target='', subset_key=theme, metadata=record)

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        cases = [
            JudgeCase(case_id=f'rubric_{index}', output_contract=RUBRIC_CONTRACT, metadata={'rubric_index': index})
            for index in range(len((context.task_state.metadata or {}).get('rubrics', [])))
        ]

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            from .utils import RubricItem
            metadata = judge_context.task_state.metadata or {}
            rubric = RubricItem.from_dict(metadata['rubrics'][case.metadata['rubric_index']])
            conversation = metadata['prompt'] + [dict(content=judge_context.original_prediction, role='assistant')]
            prompt = GRADER_TEMPLATE.replace(
                '<<conversation>>', '\n\n'.join(f"{message['role']}: {message['content']}" for message in conversation)
            ).replace('<<rubric_item>>', str(rubric))
            return JudgeRequest(messages=[ChatMessageUser(content=prompt)])

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            from .utils import RubricItem, calculate_rubric_tag_scores, calculate_score, construct_readable_explanation
            metadata = copy.deepcopy(judge_context.task_state.metadata or {})
            items = [RubricItem.from_dict(item) for item in metadata['rubrics']]
            by_case = {verdict.case_id: verdict for verdict in case_verdicts}
            responses = [by_case[f'rubric_{index}'].value.model_dump() for index in range(len(items))]
            tags, axes = calculate_rubric_tag_scores(items, responses)
            return ReducedVerdict(
                value={
                    'overall_score': calculate_score(items, responses),
                    **axes
                },
                metadata={
                    'readable_explanation': construct_readable_explanation(items, responses),
                    'rubric_tag_scores': tags,
                }
            )

        def finalize(score, review, judge_context) -> Score:
            judge_context.task_state.target = '**Score Explanation**\n\n' + review.metadata.get(
                'readable_explanation', ''
            )
            return score

        return JudgeDefinition.workflow(
            cases=cases, request=request, reduce=reduce, main_score_name='overall_score', finalize=finalize
        )
