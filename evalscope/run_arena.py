# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa

import argparse
import os
from pathlib import Path
import torch
from tqdm import tqdm

from evalscope.constants import EvalConfigKeys
from evalscope.evaluator.rating_eval import RatingEvaluate
from evalscope.models.model_adapter import ChatGenerationModelAdapter
from evalscope.utils import get_obj_from_cfg, yaml_to_dict, jsonl_to_list, dump_jsonl_data
from evalscope.utils.logger import get_logger
from modelscope.utils.hf_util import GenerationConfig

logger = get_logger()

WORK_DIR = Path(__file__).absolute().parent


class ArenaWorkflow:

    def __init__(self, cfg_file: str, **kwargs):

        self.cfg_dict = yaml_to_dict(os.path.join(WORK_DIR, cfg_file))
        logger.info(f'**Arena Config: {self.cfg_dict}')

        self.question_file: str = os.path.join(WORK_DIR, self.cfg_dict.get('question_file'))
        self.answers_gen: dict = self.cfg_dict.get('answers_gen', {})
        self.reviews_gen: dict = self.cfg_dict.get('reviews_gen', {})
        self.reviewer_cfg: dict = ArenaWorkflow._get_obj_from_cfg(self.reviews_gen.get('reviewer', {}))

        self.prompt_file = os.path.join(WORK_DIR, self.reviews_gen.get('prompt_file'))
        self.review_file = os.path.join(WORK_DIR, self.reviews_gen.get('review_file'))

        self.rating_gen: dict = self.cfg_dict.get('rating_gen', {})
        self.report_file: str = os.path.join(WORK_DIR, self.rating_gen.get('report_file'))

    @staticmethod
    def _get_obj_from_cfg(obj_cfg: dict):
        cls_ref = obj_cfg.get(EvalConfigKeys.CLASS_REF, None)
        if not cls_ref:
            logger.warning(
                f'Class reference is not specified in config: {obj_cfg}')
            return obj_cfg

        cls = get_obj_from_cfg(cls_ref)
        obj_cfg[EvalConfigKeys.CLASS_REF] = cls

        return obj_cfg

    def _predict_answers(self,
                         model_id_or_path: str,
                         model_revision: str,
                         precision: torch.dtype,
                         generation_config: GenerationConfig,
                         template_type: str) -> list:

        # TODO: multi-task to be supported
        model_adapter = ChatGenerationModelAdapter(model_id=model_id_or_path,
                                                   model_revision=model_revision,
                                                   torch_dtype=precision,
                                                   generation_config=generation_config,
                                                   template_type=template_type)
        res_list = []
        questions_list = jsonl_to_list(self.question_file)
        for data_d in tqdm(questions_list, total=len(questions_list), desc=f'Predicting(answers):'):
            # {"question_id": 1, "text": "How can I improve my time management skills?", "category": "generic"}
            text = data_d.get('text', None)
            if not text:
                logger.warning(f'Invalid question: {data_d}')
                continue
            prompt = f'Question: {text}\n\nAnswer:'
            inputs = {'data': [prompt]}
            res_d: dict = model_adapter.predict(inputs=inputs)
            ans_text: str = res_d['choices'][0]['message']['content']

            ans = {
                'question_id': data_d['question_id'],
                'text': data_d['text'],
                'category': data_d['category'],
                'model_id': model_id_or_path,
                'metadata': {},
                'answer': ans_text,
            }
            res_list.append(ans)

        return res_list

    def get_answers(self):
        for model_name, cfg_d in self.answers_gen.items():
            enable = cfg_d.get(EvalConfigKeys.ENABLE, True)
            if not enable:
                logger.warning(
                    f'Skip model {model_name} because it is not enabled.')
                continue

            model_id_or_path = cfg_d.get(EvalConfigKeys.MODEL_ID_OR_PATH)
            model_revision = cfg_d.get(EvalConfigKeys.MODEL_REVISION, None)
            precision = cfg_d.get(EvalConfigKeys.PRECISION, torch.float16)
            precision = eval(precision) if isinstance(precision, str) else precision
            custom_generation_config = cfg_d.get(EvalConfigKeys.GENERATION_CONFIG, {})
            custom_generation_config = GenerationConfig(**custom_generation_config)
            ans_output_file = os.path.join(WORK_DIR, cfg_d.get(EvalConfigKeys.OUTPUT_FILE))
            template_type = cfg_d.get(EvalConfigKeys.TEMPLATE_TYPE)

            answers_list = self._predict_answers(model_id_or_path=model_id_or_path,
                                                 model_revision=model_revision,
                                                 precision=precision,
                                                 generation_config=custom_generation_config,
                                                 template_type=template_type)

            os.makedirs(os.path.dirname(ans_output_file), exist_ok=True)
            dump_jsonl_data(answers_list, ans_output_file)
            logger.info(f'Answers generated by model {model_name} and saved to {ans_output_file}')

    def get_reviews(self, dry_run: bool = False):
        enable = self.reviews_gen.get(EvalConfigKeys.ENABLE, True)
        if enable:
            reviewer_cls = self.reviewer_cfg.get(EvalConfigKeys.CLASS_REF)
            if not reviewer_cls:
                logger.warning('Skip reviews generation because class reference is not specified.')
                return
            reviewer_args = self.reviewer_cfg.get(EvalConfigKeys.CLASS_ARGS, {})
            target_answers = self.reviews_gen.get('target_answers')
            if target_answers is None:
                # Get all answers from answers_gen config if target_answers is None
                target_answers = [item[EvalConfigKeys.OUTPUT_FILE] for item in self.answers_gen.values()]
            target_answers = [os.path.join(WORK_DIR, item) for item in target_answers]
            target_answers = [file_path for file_path in target_answers if os.path.exists(file_path)]

            baseline_file = self.reviews_gen.get('baseline_file', None)
            if baseline_file:
                baseline_file = os.path.join(WORK_DIR, baseline_file)

            reference_file = self.reviews_gen.get('reference_file', None)
            if reference_file:
                reference_file = os.path.join(WORK_DIR, reference_file)

            cache_file = self.reviews_gen.get('cache_file', None)
            if cache_file:
                cache_file = os.path.join(WORK_DIR, cache_file)

            input_kwargs = dict(
                prompt_file=self.prompt_file,
                answer_file_list=target_answers,
                review_result_file=self.review_file,
                baseline_file=baseline_file,
                reference_file=reference_file,
                reviewer_args=reviewer_args,
                cache_file=cache_file)

            reviewer_obj = reviewer_cls(**input_kwargs)
            reviewer_obj.run(dry_run=dry_run)
            logger.info(f'Reviews with generated by reviewer and saved to {self.review_file}')

        else:
            logger.warning('Skip reviews generation because it is not enabled.')

    def get_rating_results(self):
        enable = self.rating_gen.get(EvalConfigKeys.ENABLE, True)
        if enable:
            report_file = os.path.join(WORK_DIR, self.rating_gen.get('report_file'))
            metrics = self.rating_gen.get('metrics', ['elo'])
            baseline_model = self.rating_gen.get(
                'baseline_model') if metrics[0] == 'pairwise' else None
            ae = RatingEvaluate(metrics=metrics, baseline_model=baseline_model)
            res_list = ae.run(self.review_file)
            rating_df = res_list[0]
            logger.info(f'Rating results:\n{rating_df.to_csv()}')
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            rating_df.to_csv(report_file, index=True)
            logger.info(f'Rating results are saved to {report_file}')
        else:
            logger.warning('Skip rating because it is not enabled.')

    def run(self, dry_run: bool = False):

        # Get all answers
        self.get_answers()

        # Get all reviews
        self.get_reviews(dry_run=dry_run)

        # Get rating results
        self.get_rating_results()

        logger.info('*** Arena workflow is finished. ***')


def main():

    # Usage: python evalscope/run_arena.py -c /path/to/xxx_cfg_arena.yaml

    parser = argparse.ArgumentParser(description='LLMs evaluations with arena mode.')
    parser.add_argument('-c', '--cfg-file', required=True)
    parser.add_argument('--dry-run', action='store_true', default=False)
    args = parser.parse_args()

    arena_workflow = ArenaWorkflow(cfg_file=args.cfg_file)
    arena_workflow.run(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
