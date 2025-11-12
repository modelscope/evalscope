# flake8: noqa: E501
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

GRADER_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."""  # noqa: E501

GRADER_TEMPLATE = """<|User Prompt|>\n{question}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>""".strip(
)


@register_benchmark(
    BenchmarkMeta(
        name='arena_hard',
        pretty_name='ArenaHard',
        tags=[Tags.INSTRUCTION_FOLLOWING, Tags.ARENA],
        description=
        'ArenaHard is a benchmark designed to evaluate the performance of large language models in a competitive setting, '
        'where models are pitted against each other in a series of tasks to determine their relative strengths and weaknesses. '
        'It includes a set of challenging tasks that require reasoning, understanding, and generation capabilities. '
        'Currently not support `style-controlled winrate`; the official Judge model is `gpt-4-1106-preview`, while the baseline model is `gpt-4-0314`.',
        dataset_id='AI-ModelScope/arena-hard-auto-v0.1',
        metric_list=['winrate'],
        aggregation='elo',
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{question}'
    )
)
class ArenaHardAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._use_llm_judge = True  # Use LLM as a judge by default

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        question = record['question']
        baseline_prediction = record['prediction']  # baseline model prediction

        return Sample(
            input=question, target=baseline_prediction, metadata={'capability': record.get('capability', 'unknown')}
        )

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        from .utils import get_judge_score, post_process_arenahard

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        question = task_state.input_text

        # reference is baseline answer 'A', filtered_prediction is model answer 'B'
        prompt1 = GRADER_TEMPLATE.format(question=question, answer_1=reference, answer_2=filtered_prediction)
        # reverse the order
        prompt2 = GRADER_TEMPLATE.format(question=question, answer_1=filtered_prediction, answer_2=reference)

        # get grading response
        game1_response = self.llm_judge.judge(prompt1, system_prompt=GRADER_SYSTEM_PROMPT)
        game2_response = self.llm_judge.judge(prompt2, system_prompt=GRADER_SYSTEM_PROMPT)

        # parse grading response
        res1 = post_process_arenahard(game1_response)
        res2 = post_process_arenahard(game2_response)

        score1 = get_judge_score(res1, reverse=True)
        score2 = get_judge_score(res2, reverse=False)

        battle_result = {
            'model_a':
            'gpt4-0314',
            'model_b':
            'test_model',
            'games': [
                {
                    'user_prompt': prompt1,
                    'judgment': game1_response,
                    'score': res1
                },
                {
                    'user_prompt': prompt2,
                    'judgment': game2_response,
                    'score': res2
                },
            ]
        }

        # Set score based on the battle result
        score.value = {'score': (score1 + score2) / 2}
        score.explanation = f'LLM judge battles: Game1: {game1_response[:100]}... Game2: {game2_response[:100]}...'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
            'battle_result': battle_result
        }
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        import pandas as pd

        from .utils import compute_mle_elo, get_battles_from_row, get_bootstrap_result, get_win_rate_column

        battles = pd.concat([get_battles_from_row(res.score.metadata['battle_result']) for res in sample_scores])

        bootstrap_online_elo = compute_mle_elo(battles)

        stats = pd.DataFrame()
        stats['results'] = None
        stats['results'] = stats['results'].astype('object')

        for i, model in enumerate(bootstrap_online_elo.index):
            # assert model in bootstrap_elo_lu.columns
            stats.at[i, 'model'] = model
            stats.at[i, 'score'] = bootstrap_online_elo[model]

        score = get_win_rate_column(stats, 'score', 'gpt4-0314').at['test_model']

        return [AggScore(
            score=score,
            metric_name='winrate',
            num=len(sample_scores),
        )]
