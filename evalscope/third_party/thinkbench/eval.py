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
from typing import Any, Dict, List, Optional

from evalscope.third_party.thinkbench.tools.llm import request_url
from evalscope.third_party.thinkbench.tools.utils import extract_answer
from evalscope.utils.io_utils import dict_to_json, dump_jsonl_data, json_to_dict, jsonl_to_list

cur_path = os.path.dirname(os.path.abspath(__file__))


class EvalThink:

    def __init__(
        self,
        report_path: str,
        tokenizer_path: str,
        model_name: str,
        dataset_name: str,
        subsets: List[str],
        split_strategies: str = 'llm',
        judge_config: Optional[Dict[str, str]] = None,
    ):
        self.report_path = report_path
        self.reformat_template = open(os.path.join(cur_path, 'resources/reformat_template.txt'), 'r').read()
        self.critique_template = open(os.path.join(cur_path, 'resources/critique_template.txt'), 'r').read()
        self.switch_tokens = [
            'alternatively', 'but wait', 'let me reconsider',
            'another way', 'another approach', 'another method', 'another angle',
        ]
        self.subset_dict: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.think_end_token = '</think>'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.subsets = subsets
        self.metrics = [
            'reasoning_tokens', 'first_correct_tokens', 'reflection_tokens',
            'token_efficiency', 'thought_num', 'accuracy',
        ]
        self.split_strategies = split_strategies  # split by llm, keywords, separator
        self.judge_config = judge_config
        self.model_parse_file_path = os.path.join(self.report_path, 'answer_index.jsonl')
        self.model_parse_dict = self.__init_parse_file()

    def __init_parse_file(self):
        if not os.path.exists(self.model_parse_file_path):
            return {}
        else:
            list_file = jsonl_to_list(self.model_parse_file_path)
            # convert to dict prompt as key, answer_index as value
            return {item['prompt']: item['answer_index'] for item in list_file}

    def _get_assistant_message(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the last assistant message from a review item.

        Supports both old format (choices array) and new format (messages list).
        """
        # New format: messages is a list of {role, content, ...}
        if 'messages' in item:
            messages = item['messages']
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get('role') == 'assistant':
                        return msg
        # Legacy format: choices array
        if 'choices' in item:
            choices = item['choices']
            if isinstance(choices, list) and len(choices) > 0:
                return choices[0].get('message', {})
        return {}

    def _get_gold_answer(self, item: Dict[str, Any]) -> str:
        """Extract the gold/target answer from a review item."""
        # New format
        if 'target' in item and item['target'] is not None and not pd.isna(item['target']):
            return str(item['target'])
        # Legacy format
        if 'choices' in item:
            choices = item['choices']
            if isinstance(choices, list) and len(choices) > 0:
                return choices[0].get('review', {}).get('gold', '')
        return ''

    def _get_problem(self, item: Dict[str, Any]) -> str:
        """Extract the problem text from a review item."""
        # Legacy format
        if 'raw_input' in item and isinstance(item['raw_input'], dict):
            return item['raw_input'].get('question') or item['raw_input'].get('problem') or ''
        # New format: first user message content
        if 'messages' in item and isinstance(item['messages'], list):
            for msg in item['messages']:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        return content
                    # Handle list-of-blocks format
                    if isinstance(content, list):
                        parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                parts.append(block.get('text', ''))
                            elif isinstance(block, str):
                                parts.append(block)
                        return '\n'.join(parts)
        return ''

    def _get_content(self, message: Dict[str, Any]) -> str:
        """Get text content from a message, handling list-of-blocks format."""
        content = message.get('content') or ''
        if isinstance(content, list):
            # Multimodal format: list of content blocks
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get('type') == 'text':
                        parts.append(block.get('text') or '')
                    elif block.get('type') == 'reasoning':
                        parts.append(block.get('reasoning') or '')
                elif isinstance(block, str):
                    parts.append(block)
            return '\n'.join(parts)
        return content

    def get_think_part(self, message: Dict[str, Any]) -> str:
        """Extract the thinking/reasoning portion from an assistant message."""
        # Check for explicit reasoning_content field
        reasoning = message.get('reasoning_content') or ''
        if reasoning:
            return reasoning
        # Check for reasoning block in content list
        content = message.get('content', '')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'reasoning':
                    return block.get('reasoning', '')
        # Fallback: extract text before </think> token
        text = self._get_content(message)
        last_think_end = text.rfind(self.think_end_token)
        if last_think_end == -1:
            # No think token found, treat entire output as thinking
            return text
        return text[:last_think_end]

    @lru_cache(maxsize=None)
    def cal_tokens(self, text: str):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def process_item(self, item: Dict[str, Any]):
        """Process a single review item and compute thinking efficiency metrics."""
        problem = self._get_problem(item)
        assistant_msg = self._get_assistant_message(item)
        answer = self._get_gold_answer(item)

        think_part = self.get_think_part(assistant_msg)
        tokens = self.cal_tokens(think_part)
        switch_count = sum(think_part.lower().count(token) for token in self.switch_tokens)
        useful_tokens = self.cal_tokens(self.get_first_correct(think_part, problem, answer))
        reflection_tokens = tokens - useful_tokens
        score = 0 if useful_tokens == 0 else 1

        token_efficiency = useful_tokens / tokens if tokens > 0 else 0.0
        return tokens, switch_count, token_efficiency, score, useful_tokens, reflection_tokens

    def split_by_llm(self, response, problem) -> List[str]:
        response = response.replace('\n', ' ')  # remove newline characters
        prompt = self.reformat_template.format(problem=problem, response=response)
        llm_response = request_url(self.judge_config, prompt)
        if not llm_response:
            return [response]
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
                            vertical_spacing=0.1,
                            horizontal_spacing=0.1)

        metrics_order = ['reasoning_tokens', 'first_correct_tokens', 'reflection_tokens',
                         'token_efficiency', 'thought_num', 'accuracy']

        for i, metric in enumerate(metrics_order, start=1):
            y_values = [results[metric][subset] for subset in self.subsets]
            row = (i - 1) // 3 + 1
            col = (i - 1) % 3 + 1
            fig.add_trace(
                go.Scatter(x=list(range(len(self.subsets))), y=y_values,
                           mode='lines+markers',
                           name=metric.replace('_', ' ').title()),
                row=row, col=col
            )
            for j, y in enumerate(y_values):
                fig.add_annotation(
                    x=j, y=y,
                    text=f'{y:.2f}',
                    showarrow=False, yshift=10,
                    row=row, col=col
                )

        fig.update_layout(
            height=800,
            width=1200,
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
            fig.update_yaxes(title_text=metrics_order[i - 1].replace('_', ' ').title(), row=row, col=col)

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

    def filter_df(self, df: pd.DataFrame, response_len: int = 8000, count: int = 10) -> pd.DataFrame:
        """Filter rows by response token length and return the first `count` rows."""

        def is_valid_row(row: Dict[str, Any]) -> bool:
            # New format: get assistant message content token count
            if 'messages' in row and isinstance(row['messages'], list):
                for msg in reversed(row['messages']):
                    if isinstance(msg, dict) and msg.get('role') == 'assistant':
                        content = self._get_content(msg)
                        return self.cal_tokens(content) <= response_len
                return True
            # Legacy format: choices array
            if 'choices' in row and isinstance(row['choices'], list):
                return all(
                    self.cal_tokens(choice['message']['content']) <= response_len
                    for choice in row['choices']
                )
            return True

        bools = df.apply(is_valid_row, axis=1)
        return df[bools].head(count)

    def evaluate(self, output_dir: str, max_tokens: int = 8000, count: int = 50, workers: int = 128):
        """Run thinking efficiency evaluation across all subsets."""
        for subset in self.subsets:
            review_path = os.path.join(
                self.report_path, 'reviews', self.model_name, f'{self.dataset_name}_{subset}.jsonl'
            )
            if not os.path.exists(review_path):
                print(f'Warning: Review file not found for subset {subset}: {review_path}, skipping.')
                continue
            review_df = pd.read_json(review_path, lines=True)
            review_df = self.filter_df(review_df, response_len=max_tokens, count=count)

            if len(review_df) == 0:
                print(f'Warning: No valid samples found for subset {subset}, skipping.')
                continue

            results = thread_map(
                self.process_item,
                (item for _, item in review_df.iterrows()),
                desc=f'Evaluating {subset}',
                total=len(review_df),
                max_workers=workers,
            )

            all_tokens, all_thought_num, all_efficiency, all_accuracy, all_useful, all_reflection = zip(*results)

            n = len(all_tokens)
            self.subset_dict[subset]['reasoning_tokens'] = sum(all_tokens) / n
            self.subset_dict[subset]['thought_num'] = sum(all_thought_num) / n
            self.subset_dict[subset]['token_efficiency'] = sum(all_efficiency) / n
            self.subset_dict[subset]['accuracy'] = sum(all_accuracy) / n
            self.subset_dict[subset]['first_correct_tokens'] = sum(all_useful) / n
            self.subset_dict[subset]['reflection_tokens'] = sum(all_reflection) / n

        final_results = {
            metric: {subset: self.subset_dict[subset][metric] for subset in self.subsets}
            for metric in self.metrics
        }

        self.plot_metrics(final_results, output_dir)

        # save results to json
        dict_to_json(final_results, os.path.join(self.report_path, 'think_eval_results.json'))
        return final_results


def run_task(config: dict, output_dir: str = 'outputs', max_tokens: int = 8000, count: int = 50, workers: int = 128):
    """Entry point for thinking efficiency evaluation.

    Args:
        config: Dict with keys: report_path, tokenizer_path, model_name,
                dataset_name, subsets, split_strategies, judge_config.
        output_dir: Directory to save output plots.
        max_tokens: Filter out samples whose response exceeds this token count.
        count: Number of samples per subset to evaluate.
        workers: Number of parallel workers for evaluation.
    """
    evaluator = EvalThink(**config)
    results = evaluator.evaluate(output_dir, max_tokens, count, workers)
    print(results)


def combine_results(configs: List[dict], output_path: str):
    """Combine evaluation results from multiple model configs into one comparison plot.

    Args:
        configs: List of model config dicts containing model_name and report_path.
        output_path: Path to save the comparison plot image.
    """
    if not configs:
        return {}
    combined_results = defaultdict(lambda: defaultdict(dict))
    for config in configs:
        model_name = config['model_name']
        report_path = config['report_path']
        results = json_to_dict(os.path.join(report_path, 'think_eval_results.json'))
        combined_results[model_name] = results

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Reasoning Tokens', 'First Correct Tokens', 'Reflection Tokens',
            'Token Efficiency', 'Thought Num', 'Accuracy',
        ),
        shared_xaxes=True, x_title='Subsets',
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    metrics_order = [
        'reasoning_tokens', 'first_correct_tokens', 'reflection_tokens',
        'token_efficiency', 'thought_num', 'accuracy',
    ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, metric in enumerate(metrics_order, start=1):
        row = (i - 1) // 3 + 1
        col = (i - 1) % 3 + 1

        subsets = list(next(iter(combined_results.values()))[metric].keys())

        for j, (model_name, results) in enumerate(combined_results.items()):
            y_values = [results[metric][subset] for subset in subsets]

            fig.add_trace(
                go.Scatter(
                    x=subsets, y=y_values,
                    mode='lines+markers',
                    name=model_name,
                    line=dict(color=colors[j % len(colors)]),
                    showlegend=(i == 1),
                ),
                row=row, col=col,
            )

            for k, y in enumerate(y_values):
                fig.add_annotation(
                    x=subsets[k], y=y,
                    text=f'{y:.2f}',
                    showarrow=False, yshift=10,
                    font=dict(size=12, color=colors[j % len(colors)]),
                    row=row, col=col,
                )

        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=row, col=col)

    fig.update_layout(
        height=1000,
        width=1500,
        title_text='Model Comparison Across Evaluation Metrics on MATH-500',
        title=dict(font=dict(size=22)),
        font=dict(size=14),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=14),
        ),
    )

    os.makedirs(os.path.dirname(output_path) or 'outputs', exist_ok=True)
    fig.write_image(output_path)
    print(f'Model comparison plot saved to {output_path}')

    return combined_results
