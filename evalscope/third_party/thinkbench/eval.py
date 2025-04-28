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
from evalscope.utils.io_utils import dict_to_json, dump_jsonl_data, json_to_dict, jsonl_to_list

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
        self.metrics = ['reasoning_tokens', 'first_correct_tokens', 'reflection_tokens','token_efficiency', 'thought_num', 'accuracy']
        self.split_strategies = split_strategies  # split by llm, keywords, separator
        self.judge_config = judge_config
        self.model_parse_file_path = os.path.join(self.report_path, 'answer_index.jsonl')
        self.model_parse_dict = self.__init_parse_file()

    def __init_parse_file(self):
        if not os.path.exists(self.model_parse_file_path):
           return {}
        else:
            list_file =  jsonl_to_list(self.model_parse_file_path)
            # convert to dict prompt as key, answer_index as value
            return {item['prompt']: item['answer_index'] for item in list_file}

    def get_think_part(self, message: dict) -> str:
        if 'reasoning_content' in message and message['reasoning_content']:
            return message['reasoning_content']
        else:
            text = message['content']
            last_think_end = text.rfind(self.think_end_token)
            return text[:last_think_end]

    @lru_cache(maxsize=None)
    def cal_tokens(self, text: str):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def process_choice(self, choice, problem):
        think_part = self.get_think_part(choice['message'])
        answer = choice['review']['gold']
        tokens = self.cal_tokens(think_part)
        switch_count = sum(think_part.lower().count(token) for token in self.switch_tokens)
        useful_tokens = self.cal_tokens(self.get_first_correct(think_part, problem, answer))
        reflection_tokens = tokens - useful_tokens
        # score = choice['review']['result']
        score = 0 if useful_tokens == 0 else 1
        return tokens, switch_count, useful_tokens, reflection_tokens, score

    def process_item(self, item):
        problem = item['raw_input'].get('question') or item['raw_input'].get('problem') or ''
        results = []
        for choice in item['choices']:
            results.append(self.process_choice(choice, problem))
            break  # only process the first choice

        total_tokens, switch_counts, useful_tokens, reflection_tokens, scores = zip(*results)

        avg_tokens = sum(total_tokens) / len(total_tokens)
        avg_thought_num = sum(switch_counts) / len(switch_counts)
        avg_token_efficiency = sum(useful_tokens) / sum(total_tokens)
        avg_accuracy = sum(scores) / len(scores)
        avg_useful_tokens = sum(useful_tokens) / len(useful_tokens)
        avg_reflection_tokens = sum(reflection_tokens) / len(reflection_tokens)
        return avg_tokens, avg_thought_num, avg_token_efficiency, avg_accuracy, avg_useful_tokens, avg_reflection_tokens

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
        if prompt in self.model_parse_dict:
            answer_index = self.model_parse_dict[prompt]
        else:
            llm_response = request_url(self.judge_config, prompt)
            if not llm_response:
                answer_index = -1
            else:
                answer_index = extract_answer(llm_response)

            dump_jsonl_data({'prompt': prompt, 'response': llm_response, 'answer_index': answer_index},
                            self.model_parse_file_path, dump_mode='append')
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
        # Change layout to 2x3
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=('Reasoning Tokens', 'First Correct Tokens', 'Reflection Tokens',
                                          'Token Efficiency', 'Thought Num', 'Accuracy'),
                            shared_xaxes=True, x_title='Subsets',
                            vertical_spacing=0.1,  # Decrease vertical spacing between subplots
                            horizontal_spacing=0.1)  # Decrease horizontal spacing between subplots

        metrics_order = ['reasoning_tokens', 'first_correct_tokens', 'reflection_tokens',
                        'token_efficiency', 'thought_num', 'accuracy']

        for i, metric in enumerate(metrics_order, start=1):
            y_values = [results[metric][subset] for subset in self.subsets]
            # Determine row and column for 2x3 layout
            row = (i - 1) // 3 + 1
            col = (i - 1) % 3 + 1
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
            height=800,  # Adjust height for 2x3 layout
            width=1200,   # Adjust width for 2x3 layout
            title_text=f'Evaluation Metrics for {self.model_name} on {self.dataset_name}',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        for i in range(1, len(metrics_order) + 1):
            row = (i - 1) // 3 + 1
            col = (i - 1) % 3 + 1
            fig.update_xaxes(
                ticktext=self.subsets,
                tickvals=list(range(len(self.subsets))),
                row=row, col=col
            )
            fig.update_yaxes(title_text=metrics_order[i-1].replace('_', ' ').title(), row=row, col=col)

        # Update y-axis ranges
        fig.update_yaxes(range=[500, 5000], row=1, col=1)  # Reasoning Tokens
        fig.update_yaxes(range=[0, 3000], row=1, col=2)  # First Correct Tokens
        fig.update_yaxes(range=[0, 3000], row=1, col=3)  # Reflection Tokens
        fig.update_yaxes(range=[0, 1], row=2, col=1)     # Token Efficiency
        fig.update_yaxes(range=[0, 13], row=2, col=2)    # Thought Num
        fig.update_yaxes(range=[0, 1], row=2, col=3)     # Accuracy

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{self.model_name}_{self.dataset_name}_metrics.png')
        fig.write_image(output_path)
        print(f'save figure to: {output_path}')



    def filter_df(self, df, response_len: int = 8000, count: int=10):
        def is_valid_row(row):
            return all(self.cal_tokens(choice['message']['content']) <= response_len for choice in row['choices'])

        bools = df.apply(is_valid_row, axis=1)

        return df[bools].head(count)


    def evaluate(self, output_dir, max_tokens=8000, count=50, workers=128):
        for subset in self.subsets:
            review_path = os.path.join(self.report_path, 'reviews', self.model_name, f'{self.dataset_name}_{subset}.jsonl')
            review_df = pd.read_json(review_path, lines=True)

            review_df = self.filter_df(review_df, response_len=max_tokens, count=count)

            results = thread_map(
                self.process_item,
                (item for _, item in review_df.iterrows()),
                desc=f'Evaluating {subset}',
                total=len(review_df),
                max_workers=workers
            )

            avg_tokens, avg_thought_num, avg_token_efficiency, avg_accuracy, avg_useful_tokens, avg_reflection_tokens = zip(*results)

            self.subset_dict[subset]['reasoning_tokens'] = sum(avg_tokens) / len(avg_tokens)
            self.subset_dict[subset]['thought_num'] = sum(avg_thought_num) / len(avg_thought_num)
            self.subset_dict[subset]['token_efficiency'] = sum(avg_token_efficiency) / len(avg_token_efficiency)
            self.subset_dict[subset]['accuracy'] = sum(avg_accuracy) / len(avg_accuracy)
            self.subset_dict[subset]['first_correct_tokens'] = sum(avg_useful_tokens) / len(avg_useful_tokens)
            self.subset_dict[subset]['reflection_tokens'] = sum(avg_reflection_tokens) / len(avg_reflection_tokens)


        results = {metric: {subset: self.subset_dict[subset][metric] for subset in self.subsets}
                   for metric in self.metrics}

        self.plot_metrics(results, output_dir)

        # save results to json
        dict_to_json(results, os.path.join(self.report_path, f'think_eval_results.json'))
        return results

def run_task(config, output_dir='outputs', max_tokens=8000, count=50, workers=128):
    evaluator = EvalThink(**config,)
    results = evaluator.evaluate(output_dir, max_tokens, count, workers)
    print(results)

def combine_results(configs: List[dict], output_path: str):
    """
    Combine evaluation results from multiple model configs into one plot.
    All models' results for the same metric will be shown in the same subplot for easy comparison.

    Args:
        configs: List of model config dicts containing model_name and report_path
    """
    # Combine results from different runs
    combined_results = defaultdict(lambda: defaultdict(dict))
    for config in configs:
        model_name = config['model_name']
        report_path = config['report_path']
        # Results is a dict with metric as key and subset as value
        results = json_to_dict(os.path.join(report_path, f'think_eval_results.json'))
        combined_results[model_name] = results

    # Create a 2x3 subplot layout, one subplot per metric
    fig = make_subplots(rows=2, cols=3,
                       subplot_titles=('Reasoning Tokens', 'First Correct Tokens', 'Reflection Tokens',
                                     'Token Efficiency', 'Thought Num', 'Accuracy'),
                       shared_xaxes=True, x_title='Subsets',
                       vertical_spacing=0.08,  # 减小垂直间距
                       horizontal_spacing=0.05)  # 减小水平间距

    metrics_order = ['reasoning_tokens', 'first_correct_tokens', 'reflection_tokens',
                    'token_efficiency', 'thought_num', 'accuracy']

    # Assign different colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Plot each metric in a separate subplot
    for i, metric in enumerate(metrics_order, start=1):
        row = (i - 1) // 3 + 1
        col = (i - 1) % 3 + 1

        # Get subsets from first model (assuming all models have same subsets)
        subsets = list(next(iter(combined_results.values()))[metric].keys())

        # Add all models' data for this metric to the same subplot
        for j, (model_name, results) in enumerate(combined_results.items()):
            y_values = [results[metric][subset] for subset in subsets]

            fig.add_trace(
                go.Scatter(x=subsets, y=y_values,
                          mode='lines+markers',
                          name=model_name,  # Just model name since metrics are shown in subplot titles
                          line=dict(color=colors[j % len(colors)]),
                          showlegend=(i == 1)),  # Only show legend for first metric
                row=row, col=col
            )

            # Add value annotations
            for k, y in enumerate(y_values):
                fig.add_annotation(
                    x=subsets[k],
                    y=y,
                    text=f'{y:.2f}',
                    showarrow=False,
                    yshift=10,
                    font=dict(size=12, color=colors[j % len(colors)]),
                    row=row, col=col
                )

        # Update axis ranges and labels based on metric type
        # if metric == 'token_efficiency':
        #     fig.update_yaxes(range=[0.2, 0.7], row=row, col=col)
        # elif metric == 'accuracy':
        #     fig.update_yaxes(range=[0.8, 1], row=row, col=col)

        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=row, col=col)

    # Update layout
    fig.update_layout(
        height=1000,  # 增加高度
        width=1500,   # 增加宽度
        title_text=f'Model Comparison Across Evaluation Metrics on MATH-500',
        title=dict(font=dict(size=22)),  # 增大标题字号
        font=dict(size=14),  # 增大整体字号
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=14)  # 增大图例字号
        )
    )

    # Save plot
    os.makedirs('outputs', exist_ok=True)
    fig.write_image(output_path)
    print(f'Model comparison plot saved to {output_path}')

    return combined_results

judge_config = dict(
    api_key='EMPTY',
    base_url='http://0.0.0.0:8801/v1',
    model_name='Qwen2.5-72B-Instruct',
)

distill_qwen_config = dict(
    report_path = '../eval-scope/outputs/20250218_180219',
    model_name = 'DeepSeek-R1-Distill-Qwen-7B',
    tokenizer_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

math_qwen_config = dict(
    report_path = '../eval-scope/outputs/20250219_202358',
    model_name = 'Qwen2.5-Math-7B-Instruct',
    tokenizer_path = 'Qwen/Qwen2.5-Math-7B-Instruct',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

r1_config = dict(
    report_path = '../eval-scope/outputs/20250307_000404',
    model_name = 'deepseek-r1',
    tokenizer_path = 'deepseek-ai/DeepSeek-R1',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

qwq_preview_config = dict(
    report_path = '../eval-scope/outputs/20250221_105911',
    model_name = 'qwq-32b-preview',
    tokenizer_path = 'Qwen/QwQ-32B-Preview',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

qwq_config = dict(
    report_path = '../eval-scope/outputs/20250306_181550',
    model_name = 'QwQ-32B',
    tokenizer_path = 'Qwen/QwQ-32B',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

distill_qwen_32b = dict(
    report_path = '../eval-scope/outputs/20250306_235951',
    model_name = 'deepseek-r1-distill-qwen-32b',
    tokenizer_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

qwen3_32b_think = dict(
    report_path = '../eval-scope/outputs/20250428_151817',
    model_name = 'Qwen3-32B',
    tokenizer_path = 'Qwen/Qwen3-32B',
    dataset_name = 'math_500',
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    split_strategies='separator',
    judge_config=judge_config
)

if __name__ == '__main__':
    # run_task(distill_qwen_config, count=80)
    # run_task(math_qwen_config)
    # run_task(qwq_preview_config, max_tokens=20000, count=200, workers=128)
    # run_task(r1_config, max_tokens=20000, count=200, workers=128)
    # run_task(qwq_config, max_tokens=20000, count=200, workers=128)
    run_task(qwen3_32b_think, max_tokens=20000, count=200, workers=128)
    # run_task(distill_qwen_32b, max_tokens=20000, count=200, workers=128)

    # combine_results([qwq_config, r1_config, qwq_preview_config,  distill_qwen_32b], output_path='outputs/model_comparison_metrics.png')
    # combine_results([qwq_config, r1_config, distill_qwen_32b], output_path='outputs/model_comparison_metrics_3models.png')
    # combine_results([distill_qwen_config, math_qwen_config, qwq_config, r1_config, qwq_preview_config, distill_qwen_32b], output_path='outputs/model_comparison_metrics_6models.png')
    combine_results([qwq_config, r1_config, distill_qwen_32b, qwen3_32b_think], output_path='outputs/model_comparison_metrics_4models.png')
