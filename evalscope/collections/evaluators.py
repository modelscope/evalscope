import os
from collections import defaultdict
from datetime import datetime

from evalscope.benchmarks import Benchmark
from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.evaluator import Evaluator
from evalscope.models import get_local_model, initialize_model_adapter
from evalscope.utils import logger
from evalscope.utils.io_utils import OutputsStructure, jsonl_to_list


class MixEvaluator(Evaluator):

    def __init__(self, data_adapter, model_adapter, task_cfg, outputs):
        super().__init__(
            dataset_name_or_path='mixed_data',
            data_adapter=data_adapter,
            model_adapter=model_adapter,
            task_cfg=task_cfg,
            outputs=outputs)

    def evaluate(self, samples: dict, infer_cfg: dict, debug: bool):
        logger.info(f'**** Start evaluating on dataset {self.dataset_name_or_path} ****')

        reviews_score_all = {}  # {subset_name: (score, num)}
        stage_answers_dict = {}
        stage_reviews_dict = {}

        for subset_name, prompts_list in samples.items():

            answers_list: list = self.get_answers(
                subset_name=subset_name, prompts_list=prompts_list, infer_cfg=infer_cfg, debug=debug)

            stage_answers_dict[subset_name] = answers_list

            reviews_list: list = self.get_reviews(subset_name=subset_name, answers_list=answers_list, debug=debug)

            metric_res = self.compute_metrics(reviews_list=reviews_list)
            reviews_score_all[subset_name] = (metric_res, len(reviews_list))
            stage_reviews_dict[subset_name] = reviews_list

        # Generate report
        report_map = self.dump_report(reviews_score_all)

        logger.info(f'**** Evaluation finished on {self.dataset_name_or_path} ****\n')

        return report_map


class EvaluatorCollection:

    def __init__(self, task_cfg: TaskConfig, dataset):
        self.task_cfg = task_cfg
        self.dataset = dataset
        self.model = get_local_model(task_cfg)
        self.outputs = OutputsStructure(
            outputs_dir=os.path.join(self.task_cfg.work_dir,
                                     datetime.now().strftime('%Y%m%d%H%M%S')))
        self.dataset_dict = self.parse_dataset()
        self.evaluators = self.add_evaluator()

    def parse_dataset(self):
        dataset_dict = defaultdict(lambda: defaultdict(list))
        for sample in self.dataset:
            source = sample['source']
            dataset_name, subset_name = source.split('/')
            dataset_dict[dataset_name][subset_name].append(sample['prompt'])
        return dataset_dict

    def add_evaluator(self):
        evaluators = {}
        for dataset_name in self.dataset_dict.keys():
            benchmark = Benchmark.get(dataset_name)
            data_adapter = benchmark.get_data_adapter()
            model_adapter = initialize_model_adapter(self.task_cfg, benchmark.model_adapter, self.model)
            evaluators[dataset_name] = MixEvaluator(data_adapter, model_adapter, self.task_cfg, self.outputs)
        return evaluators

    def evaluate(self):
        for dataset_name, evaluator in self.evaluators.items():
            evaluator.evaluate(
                samples=self.dataset_dict[dataset_name],
                infer_cfg=self.task_cfg.generation_config,
                debug=self.task_cfg.debug)


if __name__ == '__main__':
    dataset = jsonl_to_list('outputs/mixed_data.jsonl')
    task_cfg = TaskConfig(
        model='qwen2.5',
        api_url='http://127.0.0.1:8801/v1/chat/completions',
        api_key='EMPTY',
        eval_type=EvalType.SERVICE,
    )

    evaluator_collection = EvaluatorCollection(task_cfg, dataset)
    evaluator_collection.evaluate()
