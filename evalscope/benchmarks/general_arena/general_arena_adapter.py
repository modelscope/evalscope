import glob
import os
from collections import defaultdict
from typing import Any, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType
from evalscope.metrics import Metric, mean, metric_registry
from evalscope.report import Report, ReportKey
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

GRADER_SYSTEM_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."  # noqa: E501

GRADER_TEMPLATE = "<|User Prompt|>\n{question}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>".strip(
)  # noqa: E501


@Benchmark.register(
    name='general_arena',
    pretty_name='GeneralArena',
    tags=['Custom', 'Arena'],
    description=
    'GeneralArena is a custom benchmark designed to evaluate the performance of large language models in a competitive setting, '
    'where models are pitted against each other in custom tasks to determine their relative strengths and weaknesses. You should '
    'provide the model outputs in the format of a list of dictionaries, where each dictionary contains the model name and its report path. '
    'For detailed instructions on how to use this benchmark, please refer to the [Arena User Guide](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html).',
    dataset_id='general_arena',
    metric_list=['winrate'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    system_prompt=GRADER_SYSTEM_PROMPT,
    prompt_template=GRADER_TEMPLATE,
    extra_params={
        'models': [{
            'name': 'qwen-plus',
            'report_path': 'outputs/20250627_172550/reports/qwen-plus'
        }, {
            'name': 'qwen2.5-7b',
            'report_path': 'outputs/20250627_172817/reports/qwen2.5-7b-instruct'
        }],
        'baseline':
        'qwen2.5-7b'
    })
class GeneralArenaAdapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # register metrics
        metric_registry.register(Metric(name='winrate', object=mean))

        # whether to use LLM as a judge
        self.llm_as_a_judge = True

        extra_params = kwargs.get('extra_params', {})
        self.models = extra_params.get('models', [])
        self.baseline = extra_params.get('baseline', None)

    def load(self, **kwargs):
        self._check_names()
        self._check_reports()
        self._check_datasets()
        logger.info(f'Overall datasets: {self.overall_datasets}')
        dataset_model_dict = self._load_common_datasets()
        data_dict = self._build_pair_wise_data(dataset_model_dict)
        return data_dict

    def gen_prompt(self, input_d, subset_name, few_shot_list, **kwargs):
        return self.gen_prompt_data(input_d['question'])

    def _check_names(self):
        """Check the names of the models and baseline."""
        # check duplicate models
        model_names = [model['name'] for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ValueError(f'Duplicate model names found in the models list {model_names}.')
        # check if models list is empty
        if len(self.models) < 2:
            raise ValueError('Models list must contain at least two models.')
        # check baseline model
        if self.baseline and self.baseline not in model_names:
            raise ValueError(f'Baseline model {self.baseline} not found in the models list.')
        # check if the baseline model is not set
        if not self.baseline:
            logger.warning('Baseline model is not set. Using the first model as the baseline.')
            self.baseline = self.models[0]['name']

    def _check_reports(self):
        """Check if the report paths are valid."""
        for model in self.models:
            report_path = model.get('report_path', None)
            if not report_path or not os.path.exists(report_path):
                raise ValueError(f'Report path {report_path} for model {model["name"]} does not exist.')
            reports = []
            for report_item in glob.glob(os.path.join(report_path, '*.json')):
                report = Report.from_json(report_item)
                reports.append(report)
            model['reports'] = reports

    def _check_datasets(self):
        """Check common datasets in the reports."""
        overall_datasets = set()
        for model in self.models:
            datasets = set()
            for report in model['reports']:
                report_df = report.to_dataframe()
                # get unique (dataset, subset) tuples
                unique_datasets = set(zip(report_df[ReportKey.dataset_name], report_df[ReportKey.subset_name]))
                datasets.update(unique_datasets)
            model['datasets'] = datasets
        # get overall datasets by intersecting all models' datasets
        overall_datasets = set.intersection(*[model['datasets'] for model in self.models if 'datasets' in model])
        self.overall_datasets = overall_datasets

    def _load_common_datasets(self):
        """Load common datasets from the local path."""
        from evalscope.utils import OutputsStructure, jsonl_to_list

        dataset_dict = defaultdict(dict)
        for dataset_name, subset_name in self.overall_datasets:
            for model in self.models:
                dataset_path = model['report_path'].replace(OutputsStructure.REPORTS_DIR, OutputsStructure.REVIEWS_DIR)
                dataset_file_path = os.path.join(dataset_path, f'{dataset_name}_{subset_name}.jsonl')
                if not os.path.exists(dataset_file_path):
                    raise ValueError(
                        f'Dataset {dataset_name} with subset {subset_name} not found in model {model["name"]}.')
                dataset = jsonl_to_list(dataset_file_path)
                # sort by index
                dataset.sort(key=lambda x: x.get('index'))
                dataset_dict[(dataset_name, subset_name)][model['name']] = dataset

        return dataset_dict

    def _build_pair_wise_data(self, dataset_dict):
        """Build pairwise data for the models."""
        from .utils import process_review_item

        pairwise_data = defaultdict(dict)
        for (dataset_name, subset_name), model_data in dataset_dict.items():
            if len(model_data) < 2:
                logger.warning(f'Not enough models for dataset {dataset_name} with subset {subset_name}. Skipping.')
                continue
            # create pairwise data for each model against the baseline
            model_names = list(model_data.keys())
            for name in model_names:
                if name == self.baseline:
                    continue
                pairs = []
                for model_item, baseline_item in zip(model_data[name], model_data[self.baseline]):
                    for model_choice, baseline_choice in zip(
                            process_review_item(model_item), process_review_item(baseline_item)):
                        pairs.append({
                            'question': model_choice['Question'],
                            'answer_1': model_choice['Generated'],
                            'answer_2': baseline_choice['Generated'],
                            'model_1': name,
                            'model_2': self.baseline
                        })
                pairwise_data[f'{dataset_name}&{subset_name}@{name}&{self.baseline}'][self.eval_split] = pairs

        return pairwise_data

    def llm_match(self, gold, pred, judge=None, **kwargs):
        from .utils import get_judge_score, post_process_result

        try:
            raw_input = kwargs.get('raw_input', None)
            question = raw_input['question']
            answer_1 = raw_input['answer_1']
            answer_2 = raw_input['answer_2']
            model_1 = raw_input['model_1']
            model_2 = raw_input['model_2']
        except KeyError as e:
            logger.error(f'Missing key in raw input: {e}. Raw input: {raw_input}')
            raise

        system_template = self.system_prompt
        prompt_template = self.prompt_template

        prompt1 = prompt_template.format(question=question, answer_1=answer_1, answer_2=answer_2)
        # reverse the order
        prompt2 = prompt_template.format(question=question, answer_1=answer_2, answer_2=answer_1)
        # get grading response
        game1_response = judge(prompt1, system_prompt=system_template)
        game2_response = judge(prompt2, system_prompt=system_template)
        # parse grading response
        # game1
        res1 = post_process_result(game1_response)
        score1 = get_judge_score(res1, reverse=False)
        # game2
        res2 = post_process_result(game2_response)
        score2 = get_judge_score(res2, reverse=True)
        return {
            'score':
            mean([score1, score2]),
            'games': [
                {
                    'model_a': model_1,
                    'model_b': model_2,
                    'response': game1_response,
                    'judgment': res1
                },
                {
                    'model_a': model_2,
                    'model_b': model_1,
                    'response': game2_response,
                    'judgment': res2
                },
            ]
        }

    def compute_metric(self, review_res_list: List[dict], **kwargs) -> List[dict]:
        """
        compute score of the model
        """
        import numpy as np
        import pandas as pd

        from .utils import compute_mle_elo, get_battles_from_row, get_bootstrap_result, get_win_rate_column

        if isinstance(review_res_list[0], list):
            review_res_list = [item for sublist in review_res_list for item in sublist]

        battles = pd.concat([get_battles_from_row(res) for res in review_res_list])

        bt_model_coef = compute_mle_elo(battles, baseline_model=self.baseline)

        bootstrap_model_coef = get_bootstrap_result(
            battles, func_compute_elo=compute_mle_elo, num_round=100, baseline_model=self.baseline)

        stats = pd.DataFrame()
        stats['results'] = None
        stats['results'] = stats['results'].astype('object')

        for i, model in enumerate(bt_model_coef.index):
            # assert model in bootstrap_elo_lu.columns
            stats.at[i, 'model'] = model
            stats.at[i, 'score'] = bt_model_coef[model]
            stats.at[i, 'lower'] = np.percentile(bootstrap_model_coef[model], 2.5)
            stats.at[i, 'upper'] = np.percentile(bootstrap_model_coef[model], 97.5)

        metrics_dict = {}
        metrics_dict['winrate'] = get_win_rate_column(stats, 'score', self.baseline).to_dict()
        metrics_dict['winrate_lower'] = get_win_rate_column(stats, 'lower', self.baseline).to_dict()
        metrics_dict['winrate_upper'] = get_win_rate_column(stats, 'upper', self.baseline).to_dict()

        metrics = []
        for metric_name, models in metrics_dict.items():
            for model_name, score in models.items():
                if model_name == self.baseline:
                    continue
                metrics.append({'metric_name': metric_name, 'score': score, 'num': len(review_res_list)})
        return metrics

    def post_process_report(self, report: 'Report', **kwargs):
        """Post-process the report to convert it to a DataFrame with winrate leaderboards."""
        import pandas as pd
        import tabulate

        report_path = kwargs.get('report_path')
        leaderboard_file = os.path.join(report_path, 'leaderboard.txt')

        # Ensure report directory exists
        os.makedirs(report_path, exist_ok=True)

        # Convert report to dataframe
        df = report.to_dataframe()

        # Filter for winrate-related metrics
        winrate_df = df[df[ReportKey.metric_name].str.contains('winrate')].copy()

        if winrate_df.empty:
            logger.warning('No winrate data found in the report.')
            return

        # Get all model names from self.models
        all_model_names = [model['name'] for model in self.models]

        # Collect all leaderboard outputs
        leaderboard_outputs = []

        def format_leaderboard(data_df, title):
            """Format DataFrame as leaderboard with CI."""
            # Pivot to get winrate, winrate_lower, winrate_upper as columns
            pivot_df = data_df.pivot_table(
                index=[ReportKey.model_name], columns=ReportKey.metric_name, values=ReportKey.score, aggfunc='first')

            # Add baseline model with 50% winrate
            baseline_data = {'winrate': 0.5, 'winrate_lower': 0.5, 'winrate_upper': 0.5}

            # Create a complete index with all models
            complete_index = pd.Index(all_model_names, name=pivot_df.index.name)
            pivot_df = pivot_df.reindex(complete_index)

            # Fill baseline model data
            if self.baseline in pivot_df.index:
                for col, val in baseline_data.items():
                    if col in pivot_df.columns:
                        pivot_df.loc[self.baseline, col] = val

            # Fill missing values with winrate score for other models
            if 'winrate' in pivot_df.columns:
                pivot_df['winrate_lower'] = pivot_df.get('winrate_lower', pivot_df['winrate'])
                pivot_df['winrate_upper'] = pivot_df.get('winrate_upper', pivot_df['winrate'])

            # Format for display
            leaderboard_data = []
            for model in pivot_df.index:
                if pd.isna(pivot_df.loc[model, 'winrate']):
                    continue

                score_pct = pivot_df.loc[model, 'winrate'] * 100
                lower_diff = (pivot_df.loc[model, 'winrate_lower'] - pivot_df.loc[model, 'winrate']) * 100
                upper_diff = (pivot_df.loc[model, 'winrate_upper'] - pivot_df.loc[model, 'winrate']) * 100

                leaderboard_data.append({
                    'Model': model,
                    'WinRate (%)': f'{score_pct:.1f}',
                    'CI (%)': f'({lower_diff:+.1f} / {upper_diff:+.1f})'
                })

            # Sort by score descending
            leaderboard_data.sort(key=lambda x: float(x['WinRate (%)'].replace('%', '')), reverse=True)

            # Create DataFrame
            leaderboard_df = pd.DataFrame(leaderboard_data)
            leaderboard_df.index = range(len(leaderboard_df))

            # Format as string
            table_str = tabulate.tabulate(leaderboard_df, headers='keys', showindex=False)
            output = f'{title}\n{table_str}\n'

            logger.info(f'\n{title}\n{table_str}')
            return output

        # Parse dataset and subset information from dataset_name column
        # Format: '{dataset_name}&{subset_name}@{name}&{self.baseline}'
        def parse_dataset_key(dataset_key):
            """Parse dataset key to extract dataset_name, subset_name, and model pair."""
            parts = dataset_key.split('@')

            dataset_subset = parts[0]
            model_pair = parts[1]

            dataset_name, subset_name = dataset_subset.split('&', 1)
            model_1, model_2 = model_pair.split('&', 1)

            return dataset_name, subset_name, model_1, model_2

        # Add parsed columns
        parsed_data = []
        for _, row in winrate_df.iterrows():
            dataset_name, subset_name, model_1, model_2 = parse_dataset_key(row[ReportKey.subset_name])
            if dataset_name is not None:
                parsed_data.append({
                    'dataset_name': dataset_name,
                    'subset_name': subset_name,
                    ReportKey.model_name: model_1,
                    ReportKey.metric_name: row[ReportKey.metric_name],
                    ReportKey.score: row[ReportKey.score]
                })

        if not parsed_data:
            logger.warning('No valid dataset keys found for parsing.')
            return

        parsed_df = pd.DataFrame(parsed_data)

        # 1. Overall ranking (aggregate across all datasets and subsets)
        overall_df = parsed_df.groupby([ReportKey.model_name,
                                        ReportKey.metric_name])[ReportKey.score].mean().reset_index()
        leaderboard_outputs.append(format_leaderboard(overall_df, '=== OVERALL LEADERBOARD ==='))

        # 2. Dataset-level rankings
        datasets = parsed_df['dataset_name'].unique()
        for dataset in sorted(datasets):
            dataset_df = parsed_df[parsed_df['dataset_name'] == dataset]
            dataset_agg = dataset_df.groupby([ReportKey.model_name,
                                              ReportKey.metric_name])[ReportKey.score].mean().reset_index()
            leaderboard_outputs.append(format_leaderboard(dataset_agg, f'=== DATASET LEADERBOARD: {dataset} ==='))

        # 3. Subset-level rankings
        subsets = parsed_df[['dataset_name', 'subset_name']].drop_duplicates()
        for _, subset_row in subsets.iterrows():
            dataset_name = subset_row['dataset_name']
            subset_name = subset_row['subset_name']
            subset_df = parsed_df[(parsed_df['dataset_name'] == dataset_name)
                                  & (parsed_df['subset_name'] == subset_name)]
            leaderboard_outputs.append(
                format_leaderboard(subset_df, f'=== SUBSET LEADERBOARD: {dataset_name} - {subset_name} ==='))

        # Write all leaderboard outputs to file
        with open(leaderboard_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(leaderboard_outputs))

        logger.info(f'Leaderboard results saved to: {leaderboard_file}')

    def get_gold_answer(self, input_d):
        return f"model_1: {input_d['model_1']}\n---\n" + input_d['answer_1']

    def llm_parse_pred_result(self, result, raw_input_d=None, eval_type=EvalType.CHECKPOINT):
        return f"model_2: {raw_input_d['model_2']}\n---\n" + raw_input_d['answer_2']

    def match(self, gold, pred):
        logger.warning(f'Please use LLMJudge to match the result for {self.name}')
        return
