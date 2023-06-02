# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shortuuid

from evals.utils.utils import jsonl_to_list, jsonl_dump_data
from evals.predictors.openai_gpt_predictor import OpenaiGptPredictor
from evals.utils.logger import get_logger

logger = get_logger()


class TaskEvalGpt35Completion:

    MODEL_NAME = 'gpt-3.5-turbo'

    def __init__(self, prompts: str, task_cfg: dict):

        if not task_cfg:
            task_cfg = self._get_default_cfg()
        self.task_cfg = task_cfg

        self.question_file = self.task_cfg.pop('question_file', None)
        self.ans_file = self.task_cfg.pop('ans_file', None)
        if not os.path.exists(self.ans_file):
            os.makedirs(os.path.dirname(self.ans_file), exist_ok=True)

        self.question_list = jsonl_to_list(self.question_file)
        self.gpt_predictor = OpenaiGptPredictor()

    def _get_default_cfg(self):
        default_cfg = dict(model=self.MODEL_NAME,
                           max_tokens=1024,
                           temperature=0.2)
        return default_cfg

    def run(self):
        res_list = list()

        for question in self.question_list:
            input_data = dict()
            input_data.update(self.task_cfg)
            input_data['sys_prompt'] = ''
            input_data['user_prompt'] = question['text']
            resp = self.gpt_predictor.predict(**input_data)
            if resp:
                ans_text = resp["choices"][0]["message"]["content"]
                model_id = resp["model"]
            else:
                ans_text = ''
                model_id = ''

            ans = {
                'answer_id': shortuuid.uuid(),
                'question_id': question['question_id'],
                'model_id': model_id,
                'text': ans_text,
                'metadata': {'category': question.get('category', '')}
            }
            res_list.append(ans)

        jsonl_dump_data(res_list, self.ans_file)

        return res_list


if __name__ == '__main__':

    question_file = os.path.join(os.getcwd(), '../registry/data/questions', 'question_examples.jsonl')
    ans_file = os.path.join(os.getcwd(), '../registry/output/answers', 'answer_examples.jsonl')

    in_task_cfg = dict(model=TaskEvalGpt35Completion.MODEL_NAME,
                       max_tokens=1024,
                       temperature=0.2,
                       question_file=question_file,
                       ans_file=ans_file)
    task = TaskEvalGpt35Completion(prompts='', task_cfg=in_task_cfg)
    task.run()

