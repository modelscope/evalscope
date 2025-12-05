import copy
import os
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.dataset.dataset import DatasetDict
from evalscope.api.dataset.loader import LocalDataLoader
from evalscope.api.messages.chat_message import ChatMessageUser, dict_to_chat_message
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.function_utils import retry_call
from evalscope.utils.logger import get_logger

logger = get_logger()

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
        description=
        'HealthBench: a new benchmark designed to better measure capabilities of AI systems for health. Built in partnership with 262 physicians who have practiced in 60 countries, HealthBench includes 5,000 realistic health conversations, each with a custom physician-created rubric to grade model responses.',  # noqa: E501
        dataset_id='openai-mirror/healthbench',
        subset_list=SUBSET_LIST,
        metric_list=[
            'communication_quality',
            'instruction_following',
            'accuracy',
            'context_awareness',
            'completeness',
        ],
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

    def __init__(self, *args, **kwargs):
        """
        Initialize the HealthBench adapter.

        Sets up default configuration including:
        - LLM judge evaluation
        - Dataset version selection
        - Subset reformatting
        """
        super().__init__(*args, **kwargs)

        self._use_llm_judge = True  # Use LLM as a judge by default
        self.reformat_subset = True
        self.add_aggregation_name = False
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
        # Try to load dataset from local disk
        dataset_name_or_path = self.dataset_id
        if os.path.exists(dataset_name_or_path):
            logger.info(f'Loading dataset from {dataset_name_or_path}')
            dataset_path = dataset_name_or_path
        else:
            from modelscope import dataset_snapshot_download

            # Load dataset from remote
            logger.info(f'Loading dataset from modelscope: > dataset_name: {dataset_name_or_path}')
            # download dataset snapshot
            dataset_path = dataset_snapshot_download(dataset_name_or_path, allow_file_pattern=self.version_file)

        # Create local data loader with specified parameters
        dataset = LocalDataLoader(
            data_id_or_path=dataset_path,
            split=self.eval_split,
            sample_fields=self.record_to_sample,
            subset=os.path.splitext(self.version_file)[0],  # NOTE: using hardcoded test subset
            shuffle=self.shuffle,
        ).load()

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

    def llm_match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        """
        Evaluate AI response using LLM judge against physician-created rubrics.

        Args:
            original_prediction: The AI model's original response
            filtered_prediction: Filtered/processed version of the response
            reference: Reference answer (not used in this evaluation)
            task_state: Contains metadata including rubric items

        Returns:
            Score: Contains overall score, rubric tag scores, and explanations
        """
        from .utils import (
            RubricItem,
            calculate_rubric_tag_scores,
            calculate_score,
            construct_readable_explanation,
            parse_json_to_dict,
        )

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Extract rubric items and conversation from task metadata
        example = copy.deepcopy(task_state.metadata)
        rubric_items = [RubricItem.from_dict(d) for d in example['rubrics']]
        # Construct full conversation including the AI response
        convo_with_response = example['prompt'] + [dict(content=original_prediction, role='assistant')]
        # Format conversation as readable string
        convo_str = '\n\n'.join([f"{m['role']}: {m['content']}" for m in convo_with_response])

        # Evaluate response against each rubric item using LLM judge
        grading_response_list = []
        for rubric_item in rubric_items:
            # Create judge prompt by substituting conversation and rubric item
            grader_prompt = GRADER_TEMPLATE.replace('<<conversation>>',
                                                    convo_str).replace('<<rubric_item>>', str(rubric_item))
            messages = [ChatMessageUser(content=grader_prompt)]

            def judge_func():
                grading_response = self.llm_judge.judge(messages=messages)
                grading_response_dict = parse_json_to_dict(grading_response)
                # Validate response format and extract boolean criteria_met field
                if 'criteria_met' not in grading_response_dict or not isinstance(
                    grading_response_dict['criteria_met'], bool
                ):
                    logger.warning('Grading failed due to bad JSON output, retrying...')
                    raise ValueError('Grading failed due to bad JSON output')
                return grading_response_dict

            # Retry logic for robust evaluation
            grading_result = retry_call(judge_func, retries=3, sleep_interval=1)
            grading_response_list.append(grading_result)

        # Calculate final scores and explanations
        overall_score = calculate_score(rubric_items, grading_response_list)  # Overall weighted score
        rubric_tag_scores, axis_grades = calculate_rubric_tag_scores(
            rubric_items, grading_response_list
        )  # Scores by category
        readable_explanation = construct_readable_explanation(
            rubric_items, grading_response_list
        )  # Human-readable results

        # Set score values and metadata
        score.value = {
            'overall_score': overall_score,
            **axis_grades,  # Include axis scores at top level
        }
        score.main_score_name = 'overall_score'
        score.metadata = {
            'readable_explanation': readable_explanation,
            'rubric_tag_scores': rubric_tag_scores,
        }
        # Store explanation in sample target for reference
        task_state.target = '**Score Explanation**\n\n' + readable_explanation
        return score
