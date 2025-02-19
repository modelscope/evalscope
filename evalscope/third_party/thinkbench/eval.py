import json
import os
import pandas as pd
import plotly.graph_objects as go
import re
from collections import defaultdict
from functools import lru_cache
from modelscope import AutoTokenizer
from plotly.subplots import make_subplots

from evalscope.third_party.thinkbench.tools.llm import request_qwen

cur_path = os.path.dirname(os.path.abspath(__file__))

class EvalThink:
    def __init__(self, report_path, tokenizer_path, model_name, dataset_name, subsets):
        self.report_path = report_path
        self.split_response_prompt = open(os.path.join(cur_path, 'resources/split_response_prompt.txt'), 'r').read()
        self.switch_tokens = ['alternatively', 'but wait', 'let me reconsider', 'another way', 'another approach', 'another method', 'another angle']
        self.subset_dict = defaultdict(lambda: defaultdict(list))
        self.think_end_token = '</think>'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.subsets = subsets

    @lru_cache(maxsize=None)
    def get_think_part(self, text):
        last_think_end = text.rfind(self.think_end_token)
        return text[:last_think_end].lower()

    @lru_cache(maxsize=None)
    def cal_tokens(self, text: str):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def process_choice(self, choice):
        think_part = self.get_think_part(choice['message']['content'])
        tokens = self.cal_tokens(think_part)
        switch_count = sum(think_part.count(token) for token in self.switch_tokens)
        useful_tokens = self.cal_tokens(self.get_first_correct(think_part))
        return tokens, switch_count, useful_tokens

    def process_item(self, item):
        results = [self.process_choice(choice) for choice in item['choices']]
        tokens, switch_counts, useful_tokens = zip(*results)

        avg_tokens = sum(tokens) / len(tokens)
        avg_switch_freq = sum(switch_counts) / len(switch_counts)
        avg_token_efficiency = sum(useful_tokens) / sum(tokens)

        return avg_tokens, avg_switch_freq, avg_token_efficiency

    def split_by_llm(self, problem, solution, expected_answer):
        prompt = self.split_response_prompt.format(problem=problem, solution=solution, expected_answer=expected_answer)
        llm_response = request_qwen(prompt)
        start_idx = llm_response.find('<answer>') + len('<answer>')
        end_idx = llm_response.find('</answer>')
        return json.loads(llm_response[start_idx:end_idx].strip())

    def split_by_rule(self, text):
        pattern = r'(?=\b(?:{})\b)'.format('|'.join(map(re.escape, self.switch_tokens)))
        segments = re.split(pattern, text)
        # remove empty segments
        segments = [segment.strip() for segment in segments if segment.strip()]

        return segments if segments else [text]

    def get_first_correct(self, text: str):
        text_list = self.split_by_rule(text)
        # text_list = self.split_by_llm(text)
        return text_list[0]

    def plot_metrics(self, results, output_dir):
        metrics = ['token_efficiency', 'completion_len', 'switch_freq']
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=('Token Efficiency', 'Completion Length', 'Switch Frequency'),
                            shared_xaxes=True, x_title='Subsets')


        for i, metric in enumerate(metrics, start=1):
            y_values = [results[metric][subset] for subset in self.subsets]
            fig.add_trace(
                go.Scatter(x=list(range(len(self.subsets))), y=y_values,
                           mode='lines+markers',
                           name=metric.replace('_', ' ').title()),
                row=1, col=i
            )

        fig.update_layout(
            height=500,
            width=1200,
            title_text=f'Evaluation Metrics for {self.model_name} on {self.dataset_name}',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        for i in range(1, 4):
            fig.update_xaxes(
                ticktext=self.subsets,
                tickvals=list(range(len(self.subsets))),
                row=1, col=i
            )
            fig.update_yaxes(title_text=metrics[i-1].replace('_', ' ').title(), row=1, col=i)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{self.model_name}_{self.dataset_name}_metrics.png')
        fig.write_image(output_path)
        print(f'save figure to: {output_path}')

    def evaluate(self, output_dir):
        for subset in self.subsets:
            review_path = os.path.join(self.report_path, 'reviews', self.model_name, f'{self.dataset_name}_{subset}.jsonl')
            review_df = pd.read_json(review_path, lines=True)

            results = [self.process_item(item) for _, item in review_df.iterrows()]
            avg_tokens, avg_switch_freq, avg_token_efficiency = zip(*results)

            self.subset_dict[subset]['completion_len'] = sum(avg_tokens) / len(avg_tokens)
            self.subset_dict[subset]['switch_freq'] = sum(avg_switch_freq) / len(avg_switch_freq)
            self.subset_dict[subset]['token_efficiency'] = sum(avg_token_efficiency) / len(avg_token_efficiency)

        results = {metric: {subset: self.subset_dict[subset][metric] for subset in self.subsets}
                   for metric in ['token_efficiency', 'completion_len', 'switch_freq']}

        self.plot_metrics(results, output_dir)

        return results

distill_qwen_config = dict(
    report_path = '/mnt/data/data/user/maoyunlin.myl/eval-scope/outputs/20250218_180219',
    model_name = 'DeepSeek-R1-Distill-Qwen-7B',
    tokenizer_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
)

math_qwen_config = dict(
    report_path = '/mnt/data/data/user/maoyunlin.myl/eval-scope/outputs/20250218_180219',
    model_name = 'Qwen2.5-Math-7B-Instruct',
    tokenizer_path = 'Qwen/Qwen2.5-Math-7B-Instruct',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
)

if __name__ == '__main__':
    evaluator = EvalThink(**distill_qwen_config)
    results = evaluator.evaluate('outputs')
    print(results)
