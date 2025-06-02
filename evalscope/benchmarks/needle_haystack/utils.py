import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
import string
from matplotlib.colors import LinearSegmentedColormap


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


def parse_score(score_str: str) -> int:
    """
    Parses a score string and returns an integer score.
    The score should be in the format [[score]].
    """
    score_match = re.search(r'\[\[(\d+)\]\]', score_str)
    if score_match:
        score = int(score_match.group(1))
        return score / 10.0
    else:
        return 0.0


def draw_score_chat(pivot_table, outpath):
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#F0496E', '#EBB839', '#0CD79F'])

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(pivot_table, vmin=0.0, vmax=1.0, annot=True, fmt='.1f', cmap=cmap, cbar_kws={'label': 'Score'})

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

Only respond with a numberical score with formatted as [[score]]."""  # noqa: E501

ORM_USER_TEMPLATE = """
Question: {question}

Reference Answer: {gold}

Model Answer: {pred}
"""
