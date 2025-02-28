import json
import os
import pandas as pd
import plotly.graph_objects as go
import re
from collections import defaultdict
from functools import lru_cache
from modelscope import AutoTokenizer
from plotly.subplots import make_subplots
from tqdm.contrib.concurrent import thread_map
from typing import List

from evalscope.third_party.thinkbench.tools.llm import request_url
from evalscope.third_party.thinkbench.tools.utils import extract_answer
from evalscope.utils.io_utils import dump_jsonl_data

cur_path = os.path.dirname(os.path.abspath(__file__))

class EvalThink:
    def __init__(self, report_path, tokenizer_path, model_name, dataset_name, subsets, split_strategies='llm', judge_config=None):
        self.report_path = report_path
        self.reformat_template = open(os.path.join(cur_path, 'resources/reformat_template.txt'), 'r').read()
        self.critique_template = open(os.path.join(cur_path, 'resources/critique_template.txt'), 'r').read()
        self.switch_tokens = ['alternatively', 'but wait', 'let me reconsider', 'another way', 'another approach', 'another method', 'another angle']
        self.subset_dict = defaultdict(lambda: defaultdict(list))
        self.think_end_token = '</think>'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.subsets = subsets
        self.metrics = ['token_efficiency', 'completion_len', 'thought_num', 'accuracy']
        self.split_strategies = split_strategies  # split by llm, keywords, separator
        self.judge_config = judge_config

    @lru_cache(maxsize=None)
    def get_think_part(self, text):
        last_think_end = text.rfind(self.think_end_token)
        return text[:last_think_end].lower()

    @lru_cache(maxsize=None)
    def cal_tokens(self, text: str):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def process_choice(self, choice, problem):
        think_part = self.get_think_part(choice['message']['content'])
        answer = choice['review']['gold']
        tokens = self.cal_tokens(think_part)
        switch_count = sum(think_part.count(token) for token in self.switch_tokens)
        useful_tokens = self.cal_tokens(self.get_first_correct(think_part, problem, answer))
        score = choice['review']['result']
        return tokens, switch_count, useful_tokens, score

    def process_item(self, item):
        problem = item['raw_input'].get('question') or item['raw_input'].get('problem') or ''
        results = []
        for choice in item['choices']:
            results.append(self.process_choice(choice, problem))
            break  # only process the first choice

        tokens, switch_counts, useful_tokens, scores = zip(*results)

        avg_tokens = sum(tokens) / len(tokens)
        avg_thought_num = sum(switch_counts) / len(switch_counts)
        avg_token_efficiency = sum(useful_tokens) / sum(tokens)
        avg_accuracy = sum(scores) / len(scores)

        return avg_tokens, avg_thought_num, avg_token_efficiency, avg_accuracy

    def split_by_llm(self, response, problem) -> List[str]:
        response = response.replace('\n', ' ') # remove newline characters
        prompt = self.reformat_template.format(problem=problem, response=response)
        llm_response = request_url(self.judge_config, prompt)
        return llm_response.split('\n\n')

    def split_by_keywords(self, text) -> List[str]:
        pattern = r'(?=\b(?:{})\b)'.format('|'.join(map(re.escape, self.switch_tokens)))
        segments = re.split(pattern, text)
        # remove empty segments
        segments = [segment.strip() for segment in segments if segment.strip()]

        return segments if segments else [text]

    def split_by_separator(self, text) -> List[str]:
        return text.split('\n\n')

    def get_answer_index(self, response: List[str], problem: str, answer: str) -> int:
        tagged_response = ''
        for sdx, step in enumerate(response):
            tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
        tagged_response = tagged_response.strip()

        prompt = self.critique_template.format(problem=problem, answer=answer, tagged_response=tagged_response)
        llm_response = request_url(self.judge_config, prompt)
        answer_index = extract_answer(llm_response)

        dump_jsonl_data({'prompt': prompt, 'response': llm_response, 'answer_index': answer_index},
                        os.path.join(self.report_path, 'answer_index.jsonl'),
                        dump_mode='append')
        try:
            answer_index = int(answer_index)
        except Exception:
            answer_index = -1
        return answer_index

    def get_first_correct(self, response: str, problem: str, answer: str) -> str:
        if self.split_strategies == 'llm':
            text_list = self.split_by_llm(response, problem)
        elif self.split_strategies == 'keywords':
            text_list = self.split_by_keywords(response)
        else:
            text_list = self.split_by_separator(response)

        answer_index = self.get_answer_index(text_list, problem, answer)

        if answer_index == -1:  # no correct answer found
            first_correct = ''
        else:
            first_correct = '\n\n'.join(text_list[: answer_index])
        return first_correct

    def plot_metrics(self, results, output_dir):
        # Change layout to 2x2
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Token Efficiency', 'Completion Length', 'Thought Num', 'Accuracy'),
                            shared_xaxes=True, x_title='Subsets',
                            vertical_spacing=0.1,  # Decrease vertical spacing between subplots
                            horizontal_spacing=0.1)  # Decrease horizontal spacing between subplots

        for i, metric in enumerate(self.metrics, start=1):
            y_values = [results[metric][subset] for subset in self.subsets]
            # Determine row and column for 2x2 layout
            row = (i - 1) // 2 + 1
            col = (i - 1) % 2 + 1
            fig.add_trace(
                go.Scatter(x=list(range(len(self.subsets))), y=y_values,
                           mode='lines+markers',
                           name=metric.replace('_', ' ').title()),
                row=row, col=col
            )
            # Add annotations for each data point
            for j, y in enumerate(y_values):
                fig.add_annotation(
                    x=j,
                    y=y,
                    text=f'{y:.2f}',
                    showarrow=False,
                    yshift=10,
                    row=row,
                    col=col
                )

        fig.update_layout(
            height=800,  # Adjust height for 2x2 layout
            width=800,   # Adjust width for 2x2 layout
            title_text=f'Evaluation Metrics for {self.model_name} on {self.dataset_name}',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        for i in range(1, len(self.metrics) + 1):
            row = (i - 1) // 2 + 1
            col = (i - 1) % 2 + 1
            fig.update_xaxes(
                ticktext=self.subsets,
                tickvals=list(range(len(self.subsets))),
                row=row, col=col
            )
            fig.update_yaxes(title_text=self.metrics[i-1].replace('_', ' ').title(), row=row, col=col)
        # Update y-axis ranges
        fig.update_yaxes(range=[0, 1], row=1, col=1)  # Token Efficiency
        fig.update_yaxes(range=[0, 13], row=2, col=1)  # Switch Frequency
        fig.update_yaxes(range=[0, 1], row=2, col=2)  # Accuracy

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{self.model_name}_{self.dataset_name}_metrics.png')
        fig.write_image(output_path)
        print(f'save figure to: {output_path}')



    def filter_df(self, df, response_len: int = 8000, count: int=10):
        def is_valid_row(row):
            return all(self.cal_tokens(choice['message']['content']) <= response_len for choice in row['choices'])

        bools = df.apply(is_valid_row, axis=1)

        return df[bools].head(count)


    def evaluate(self, output_dir, max_tokens=8000, count=50):
        for subset in self.subsets:
            review_path = os.path.join(self.report_path, 'reviews', self.model_name, f'{self.dataset_name}_{subset}.jsonl')
            review_df = pd.read_json(review_path, lines=True)

            review_df = self.filter_df(review_df, response_len=max_tokens, count=count)

            results = thread_map(
                self.process_item,
                (item for _, item in review_df.iterrows()),
                desc=f'Evaluating {subset}',
                total=len(review_df),
                max_workers=16
            )

            avg_tokens, avg_thought_num, avg_token_efficiency, avg_accuracy = zip(*results)

            self.subset_dict[subset]['completion_len'] = sum(avg_tokens) / len(avg_tokens)
            self.subset_dict[subset]['thought_num'] = sum(avg_thought_num) / len(avg_thought_num)
            self.subset_dict[subset]['token_efficiency'] = sum(avg_token_efficiency) / len(avg_token_efficiency)
            self.subset_dict[subset]['accuracy'] = sum(avg_accuracy) / len(avg_accuracy)


        results = {metric: {subset: self.subset_dict[subset][metric] for subset in self.subsets}
                   for metric in self.metrics}

        self.plot_metrics(results, output_dir)

        return results

def run_task(config, output_dir='outputs', max_tokens=8000, count=50):
    evaluator = EvalThink(**config,)
    results = evaluator.evaluate(output_dir, max_tokens, count)
    print(results)

judge_config = dict(
    api_key='EMPTY',
    base_url='http://0.0.0.0:8801/v1',
    model_name='Qwen2.5-72B-Instruct',
)

distill_qwen_config = dict(
    report_path = '/mnt/data/data/user/maoyunlin.myl/eval-scope/outputs/20250218_180219',
    model_name = 'DeepSeek-R1-Distill-Qwen-7B',
    tokenizer_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

math_qwen_config = dict(
    report_path = '/mnt/data/data/user/maoyunlin.myl/eval-scope/outputs/20250219_202358',
    model_name = 'Qwen2.5-Math-7B-Instruct',
    tokenizer_path = 'Qwen/Qwen2.5-Math-7B-Instruct',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

r1_config = dict(
    report_path = '/mnt/data/data/user/maoyunlin.myl/eval-scope/outputs/20250221_104202',
    model_name = 'deepseek-r1',
    tokenizer_path = 'deepseek-ai/DeepSeek-R1',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

qwq_config = dict(
    report_path = '/mnt/data/data/user/maoyunlin.myl/eval-scope/outputs/20250221_105911',
    model_name = 'qwq-32b-preview',
    tokenizer_path = 'Qwen/QwQ-32B-Preview',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

if __name__ == '__main__':
    run_task(distill_qwen_config, count=80)
    # run_task(math_qwen_config)
    # run_task(r1_config)
    # run_task(qwq_config, count=80)
