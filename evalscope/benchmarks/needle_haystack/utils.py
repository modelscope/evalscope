import os
import re
import string
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from evalscope.report import Report


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def collect_cell_scores(report: 'Report') -> 'pd.DataFrame':
    """Collect the mean accuracy of every (context length, depth) cell in a report, per subset.

    Cells are read from the ``context_length`` and ``depth_percent`` dimensions of each metric
    identity, never from the display name. Reports written before the metric semantics contract
    are migrated to the same dimensions when they are loaded.

    Args:
        report: Needle-in-a-Haystack report.

    Returns:
        One row per subset and cell, with ``Subset``, ``Context``, ``Depth`` and ``Score`` columns.
    """
    import pandas as pd

    rows = []
    for metric in report.metrics:
        identity = metric.identity
        if (
            identity.name != 'accuracy'
            or identity.aggregation != 'mean'
            or not {'context_length', 'depth_percent'} <= identity.dimensions.keys()
        ):
            continue
        for category in metric.categories:
            for subset in category.subsets:
                if subset.is_aggregate:
                    continue
                rows.append(
                    {
                        'Subset': subset.name,
                        'Context': identity.dimensions['context_length'],
                        'Depth': identity.dimensions['depth_percent'],
                        'Score': subset.score,
                    }
                )
    return pd.DataFrame(rows, columns=['Subset', 'Context', 'Depth', 'Score'])


def draw_score_chat(pivot_table, outpath, show_score=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#F0496E', '#EBB839', '#0CD79F'])

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(pivot_table, vmin=0.0, vmax=1.0, annot=show_score, fmt='.1f', cmap=cmap, cbar_kws={'label': 'Score'})

    # More aesthetics
    plt.title('Fact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # save the figure
    plt.savefig(outpath, dpi=300, bbox_inches='tight')


GENERAL_ORM_PROMPT = """You are an expert in verifying if the model answer is correct based on the reference answer.
Your input is a question, a reference answer, and a model answer. You need to check if the model answer is correct based on the reference answer.
You should focus on the correctness of the model answer compared to the reference answer, without attempting to solve the original question.
You must provide your final score in the form of a number from 1 to 10, where:

Score 1: The answer is completely unrelated to the reference.
Score 3: The answer has minor relevance but does not align with the reference.
Score 5: The answer has moderate relevance but contains inaccuracies.
Score 7: The answer aligns with the reference but has minor omissions.
Score 10: The answer is completely accurate and aligns perfectly with the reference.
"""  # noqa: E501

ORM_USER_TEMPLATE = """
Question: {question}

Reference Answer: {gold}

Model Answer: {pred}
"""
