import re
from collections import defaultdict
from typing import Any, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import LLMJudge, Metric, mean, metric_registry
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

GRADER_SYSTEM_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."  # noqa: E501

GRADER_TEMPLATE = "<|User Prompt|>\n{question}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>".strip(
)  # noqa: E501


@Benchmark.register(
    name='arena_hard',
    pretty_name='ArenaHard',
    dataset_id='AI-ModelScope/arena-hard-auto-v0.1',
    metric_list=['winrate'],
    few_shot_num=0,
    train_split=None,
    eval_split='test')
class AlpacaEvalAdapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # register metrics
        metric_registry.register(Metric(name='winrate', object=mean))

        # whether to use LLM as a judge
        self.llm_as_a_judge = True

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        question = input_d['question']
        return self.gen_prompt_data(question)

    def get_gold_answer(self, input_d: dict) -> str:
        return input_d['prediction']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, **kwargs) -> str:
        return result.strip()

    def match(self, gold: str, pred: str):
        # simple match
        logger.warning(f'Please use LLMJudge to match the result for {self.name}')
        return None

    def llm_match(self, gold: Any, pred: Any, judge: LLMJudge, **kwargs) -> dict:
        from .utils import post_process_arenahard

        raw_input = kwargs.get('raw_input', None)
        question = raw_input['question']
        # gold is baseline answer 'A', pred is model answer 'B'
        prompt1 = GRADER_TEMPLATE.format(question=question, answer_1=gold, answer_2=pred)
        # reverse the order
        prompt2 = GRADER_TEMPLATE.format(question=question, answer_1=pred, answer_2=gold)
        # get grading response
        game1_response = judge(prompt1, system_prompt=GRADER_SYSTEM_PROMPT)
        game2_response = judge(prompt2, system_prompt=GRADER_SYSTEM_PROMPT)
        # parse grading response
        res1 = post_process_arenahard(game1_response)
        res2 = post_process_arenahard(game2_response)
        return {
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

    def compute_metric(self, review_res_list: List[dict], **kwargs) -> List[dict]:
        """
        compute score of the model
        """
        import pandas as pd

        from .utils import compute_mle_elo, get_battles_from_row, get_bootstrap_result, get_win_rate_column

        if isinstance(review_res_list[0], list):
            review_res_list = [item for sublist in review_res_list for item in sublist]

        battles = pd.concat([get_battles_from_row(res) for res in review_res_list])

        bootstrap_online_elo = compute_mle_elo(battles)

        # bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, 100)
        stats = pd.DataFrame()
        stats['results'] = None
        stats['results'] = stats['results'].astype('object')

        for i, model in enumerate(bootstrap_online_elo.index):
            # assert model in bootstrap_elo_lu.columns
            stats.at[i, 'model'] = model
            stats.at[i, 'score'] = bootstrap_online_elo[model]
            # stats.at[i, "lower"] = np.percentile(bootstrap_elo_lu[model], 2.5)
            # stats.at[i, "upper"] = np.percentile(bootstrap_elo_lu[model], 97.5)

        # stats['score'] = get_win_rate_column(stats, 'score', 'gpt4-0314').tolist()

        score = get_win_rate_column(stats, 'score', 'gpt4-0314').at['test_model']

        return [{'metric_name': 'winrate', 'score': score, 'num': len(review_res_list)}]
