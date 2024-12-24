import json
import os
import pandas as pd
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from evalscope.benchmarks import Benchmark
from evalscope.config import TaskConfig
from evalscope.constants import AnswerKeys, DumpMode, EvalType, ReviewKeys
from evalscope.evaluator import Evaluator
from evalscope.models import get_local_model, initialize_model_adapter
from evalscope.utils.io_utils import OutputsStructure, dump_jsonl_data, jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()


class SimpleEvaluator(Evaluator):

    def __init__(self, dataset_name, data_adapter, model_adapter, task_cfg, outputs):
        super().__init__(
            dataset_name_or_path=dataset_name,
            data_adapter=data_adapter,
            model_adapter=model_adapter,
            task_cfg=task_cfg,
            outputs=outputs)

    def get_answer(self, input_prompt, subset_name, infer_cfg) -> dict:
        answer_d: dict = self.model_adapter.predict(inputs=input_prompt, infer_cfg=infer_cfg)
        answer_id = self._generate_answer_id(self.model_adapter.model_cfg, input_prompt, infer_cfg)
        processed_answer = self._process_answer(answer_d, input_prompt, subset_name, answer_id)
        return processed_answer

    def get_review(self, answer_d) -> dict:
        review_id, reviewer_spec = self._generate_review_id(answer_d)
        review_d = self._get_review(answer_d=answer_d, review_id=review_id, reviewer_spec=reviewer_spec)
        return review_d


class EvaluatorCollection:

    def __init__(self, task_cfg: TaskConfig):
        self.task_cfg = task_cfg
        self.model = get_local_model(task_cfg)
        self.outputs = OutputsStructure(
            outputs_dir=os.path.join(self.task_cfg.work_dir,
                                     datetime.now().strftime('%Y%m%d%H%M%S')))
        self.raw_dataset = jsonl_to_list(self.task_cfg.dataset_args['data_collection']['local_path'])
        self.dataset_name_map, self.dataset_id_map = self._parse_dataset()
        self.evaluators = self._initialize_evaluators()

    def _parse_dataset(self):
        dataset_name_map = defaultdict(lambda: defaultdict(list))
        dataset_id_map = {}
        for sample in self.raw_dataset:
            dataset_name, subset_name = sample['dataset_name'], sample['subset_name']
            dataset_name_map[dataset_name][subset_name].append(sample['id'])
            dataset_id_map[sample['id']] = sample
        return dataset_name_map, dataset_id_map

    def _initialize_evaluators(self):
        evaluators = {}
        for dataset_name in self.dataset_name_map.keys():
            benchmark = Benchmark.get(dataset_name)
            data_adapter = benchmark.get_data_adapter()
            model_adapter = initialize_model_adapter(self.task_cfg, benchmark.model_adapter, self.model)
            evaluators[dataset_name] = SimpleEvaluator(dataset_name, data_adapter, model_adapter, self.task_cfg,
                                                       self.outputs)
        return evaluators

    def get_report(self, reviews):
        data = []
        for dataset_name, data_map in self.dataset_name_map.items():
            for subset_name, ids in data_map.items():
                for _id in ids:
                    review_d = reviews[_id]
                    row_data = self.dataset_id_map[_id]
                    score = self.get_pred_score(review_d)
                    data.append({
                        'task_type': row_data['task'],
                        'dataset_name': dataset_name,
                        'subset_name': subset_name,
                        'tags': row_data['tags'],
                        'score': score
                    })

        df = pd.DataFrame(data)

        # Multi-level aggregation
        subset_report_df = df.groupby(['task_type', 'dataset_name', 'subset_name']).agg(
            average_score=('score', 'mean'), count=('score', 'size')).reset_index()

        dataset_report_df = df.groupby(['task_type', 'dataset_name']).agg(
            average_score=('score', 'mean'), count=('score', 'size')).reset_index()

        task_report_df = df.groupby(['task_type']).agg(
            average_score=('score', 'mean'), count=('score', 'size')).reset_index()

        # Combine all reports into a single dictionary
        report = {
            'subset_level': subset_report_df.to_dict(orient='records'),
            'dataset_level': dataset_report_df.to_dict(orient='records'),
            'task_level': task_report_df.to_dict(orient='records')
        }

        # Log the report
        logger.info(f"Report:\n{pd.DataFrame(report['subset_level']).to_markdown(index=False)}")

        # Save the report to a JSON file
        report_file_path = os.path.join(self.outputs.reports_dir, 'data_collection.json')
        with open(report_file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

    def get_answers(self):
        pred_file_path = os.path.join(self.outputs.predictions_dir, 'data_collection.jsonl')
        answers = defaultdict(dict)
        for sample in tqdm(self.raw_dataset, desc='Getting answers'):
            evaluator = self.evaluators[sample['dataset_name']]
            answer_d = evaluator.get_answer(sample['prompt'], sample['subset_name'], self.task_cfg.generation_config)
            answers[sample['id']] = answer_d
            dump_jsonl_data(answer_d, pred_file_path, dump_mode=DumpMode.APPEND)
        return answers

    def get_reviews(self, answers):
        review_file_path = os.path.join(self.outputs.reviews_dir, 'data_collection.jsonl')
        reviews = defaultdict(dict)
        for sample in tqdm(self.raw_dataset, desc='Getting reviews'):
            evaluator = self.evaluators[sample['dataset_name']]
            review_d = evaluator.get_review(answers[sample['id']])
            reviews[sample['id']] = review_d
            dump_jsonl_data(review_d, review_file_path, dump_mode=DumpMode.APPEND)
        return reviews

    @staticmethod
    def get_pred_score(review_d) -> float:
        return review_d[AnswerKeys.CHOICES][0][ReviewKeys.REVIEW][ReviewKeys.RESULT]

    def evaluate(self):
        answers = self.get_answers()
        reviews = self.get_reviews(answers)
        self.get_report(reviews)


if __name__ == '__main__':
    task_cfg = TaskConfig(
        model='qwen2.5',
        api_url='http://127.0.0.1:8801/v1/chat/completions',
        api_key='EMPTY',
        eval_type=EvalType.SERVICE,
        datasets=['data_collection'],
        dataset_args={'data_collection': {
            'local_path': 'outputs/mixed_data.jsonl'
        }},
    )

    evaluator_collection = EvaluatorCollection(task_cfg)
    evaluator_collection.evaluate()
