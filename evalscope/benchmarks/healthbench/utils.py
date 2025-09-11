import json
import re
from collections import defaultdict

from evalscope.utils import get_logger

logger = get_logger()


def parse_json_to_dict(json_string: str) -> dict:
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r'^```json\s*|\s*```$', '', json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f'JSON decoding failed: {e}')
        return {}


class RubricItem:

    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f'[{self.points}] {self.criterion}'

    def to_dict(self):
        return {
            'criterion': self.criterion,
            'points': self.points,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d['criterion'],
            points=d['points'],
            tags=d['tags'],
        )


def calculate_score(rubric_items: list[RubricItem], grading_response_list: list[dict]) -> float | None:
    total_possible_points = sum(rubric_item.points for rubric_item in rubric_items if rubric_item.points > 0)
    if total_possible_points == 0:
        # should not happen for overall score, but may happen for tags
        return None

    achieved_points = sum(
        rubric_item.points
        for rubric_item, grading_response in zip(rubric_items, grading_response_list, strict=True)
        if grading_response['criteria_met']
    )
    overall_score = achieved_points / total_possible_points
    return overall_score


def calculate_rubric_tag_scores(rubric_items: list[RubricItem], grading_response_list: list[dict]) -> dict[str, float]:
    rubric_tag_items_grades = defaultdict(list)
    axis_grades = defaultdict(list)
    for rubric_item, grading_response in zip(rubric_items, grading_response_list):
        curr_item_tags = set()  # Ensure no duplicates in a rubric item.
        for tag in rubric_item.tags:
            rubric_tag_items_grades[tag].append((rubric_item, grading_response))
            assert tag not in curr_item_tags
            curr_item_tags.add(tag)

    rubric_tag_scores = {}
    for tag, items_grades in rubric_tag_items_grades.items():
        items, grades = zip(*items_grades)
        score = calculate_score(items, grades)
        if score is not None:  # implies at least one positive criterion
            rubric_tag_scores[tag] = score
            if tag.startswith('axis:'):
                axis_grades[tag.split(':')[1]] = score

    return rubric_tag_scores, axis_grades


def construct_readable_explanation(rubric_items: list[RubricItem], grading_response_list: list[dict]) -> str:
    rubric_items_with_grades = []
    readable_explanation_list = []
    for rubric_item, grading_response in zip(rubric_items, grading_response_list):
        explanation = grading_response.get('explanation', 'No explanation provided')
        criteria_met = grading_response['criteria_met']
        readable_explanation = (f'[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}')
        readable_explanation_list.append(readable_explanation)
        rubric_items_with_grades.append({
            **rubric_item.to_dict(),
            'criteria_met': criteria_met,
            'explanation': explanation,
        })

    readable_explanation_list.sort(key=lambda x: x.startswith('[False]'), reverse=True)
    readable_explanation_str = '\n\n'.join(readable_explanation_list)
    readable_explanation_str = f'\n\n{readable_explanation_str}'

    return readable_explanation_str
