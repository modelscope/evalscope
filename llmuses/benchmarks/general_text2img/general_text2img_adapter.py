# Copyright (c) Alibaba, Inc. and its affiliates.

from llmuses.benchmarks.data_adapter import DataAdapter
from llmuses.metrics.metrics import bleu_ngram_one_sample, weighted_mean
from llmuses.metrics.rouge_metric import compute_rouge_score_one_sample_zh
from llmuses.metrics.multi_modal_metric import fid_score, is_score, clip_score
from llmuses.utils.logger import get_logger
from typing import Any, Optional
from collections import defaultdict
import json
import cv2
import csv
import urllib.request
import numpy as np
import torch

logger = get_logger()

DATASET_ID = 'general_text2img'
SUBSET_LIST = ['default']


class GeneralText2ImgAdapter(DataAdapter):

    def __init__(self,
                 subset_list: list = None,
                 metric_list: list = None,
                 train_split: str = None,
                 eval_split: str = 'validation',
                 **kwargs):
        if subset_list is None:
            subset_list = SUBSET_LIST

        if metric_list is None:
            metric_list = [{'name': 'FID', 'object': weighted_mean}]
        
        super().__init__(subset_list=subset_list,
                         metric_list=metric_list,
                         train_split=train_split,
                         eval_split=eval_split,
                         **kwargs)
    
    def load(self,
            dataset_name_or_path: str,
            subset_list: list = None,
            work_dir: str = "",
            **kwargs) -> dict:
        data_dict = {}

        split_list = [split for split in [self.train_split, self.eval_split] if split is not None]
        for sub_name in subset_list:
            data_dict[sub_name] = {}

            try:
                with open(dataset_name_or_path, encoding="utf-8") as f:
                    rows = []
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if len(row) != 5:
                            print(f"Mismatch len of row: {row}, len of row should be 6. Skip this row.")
                            continue
                        rows.append({
                            "uniq_id": row[1],
                            "caption": row[3],
                            "image": row[4]
                        })
            except Exception as e:
                raise e
            
            for split in split_list:
                dataset = rows
                data_dict[sub_name].update({split: dataset})

        return data_dict
    
    def gen_prompt(self, input_d: list, subset_name: str, few_shot_list: list, user_prompt: dict, **kwargs) -> dict:
        """
        Args:
            input_d: [{'question': '', 'answer': ''},{'question': '', 'answer': ''},...]

        Returns:
            {'data': [prompt]}

        """
        # prompt = f"'<|im_start|>user\n{input_d['input']}<|im_end|>\n<|im_start|>assistant\n'"
        prompt = input_d["caption"]
        uniq_id = input_d["uniq_id"]

        return {'data': [prompt], "id": uniq_id}
    
    def get_gold_answer(self, input_d: list) -> str:
        """
        Args:
            input_d: [{'question': '', 'answer': ''},{'question': '', 'answer': ''},...]

        Returns:
            gold_answer: str

        """
        return input_d.get('image', '')
    
    def parse_pred_result(self, result: list, raw_input_d: dict = None) -> str:
        """
        Args:
            result: str

        Returns:
            pred_result: str

        """
        caption = raw_input_d.get("caption", "")
        return {"pred": result, "caption": caption}
    
    def match(self, gold: str, pred: list) -> float:
        """
        Args:
            gold: str
            pred: str

        Returns:
            bleu_score: float

        """
        # get gold label img
        res = urllib.request.urlopen(gold)
        i = np.asarray(bytearray(res.read()), dtype="uint8")
        label_img_arr = cv2.imdecode(i, cv2.IMREAD_COLOR)
        label_img_tensor = torch.tensor(label_img_arr, dtype=torch.uint8)
        label_img_tensor = label_img_tensor.permute(2, 0, 1).unsqueeze(0)
        label_imgs_tensor = torch.cat((label_img_tensor, label_img_tensor), 0)

        # get pred imgs
        pred_imgs_arr = [cv2.imread(img_path) for img_path in pred.get("pred", [])]
        pred_imgs_tensor = torch.tensor(pred_imgs_arr, dtype=torch.uint8)
        pred_imgs_tensor = pred_imgs_tensor.permute(0, 3, 1, 2)

        # get prompt
        prompt = pred.get("caption", "")

        fid = fid_score(label_imgs_tensor, pred_imgs_tensor)
        is_mean, is_std = is_score(pred_imgs_tensor)
        clipscore = clip_score(pred_imgs_tensor[0], prompt)
        res = {
            "fid": fid,
            "is_mean": is_mean,
            "is_std": is_std,
            "clip_score": clipscore
        }

        return res
    
    def compute_metric(self, review_res_list: list) -> float:
        """
        compute weighted mean of the bleu score of all samples

        Args:
            review_res_list: [score1, score2, ...]

        Returns:
            avg_res: float

        """
        items = defaultdict(list)
        for scores in review_res_list:
            for k,v in scores.items():
                items[k].append((v, 1.0))
        # items = [(score, 1.0) for score in review_res_list]
        res = {k: weighted_mean(v) for k,v in items.items()}
        # return weighted_mean(items)
        return res
    
    def gen_report(self, subset_score_map: dict) -> dict:
        """
        Args:
            subset_score_map: {subset_name: (score_dict, num), ...}

        Returns:
        {
            "name":"COCOTest",
            "metric":"FID",
            "score":0.399,
            "category":[
                {
                    "name":"DEFAULT",
                    "score":0.399,
                    "subset":[
                        {
                            "name":"default",
                            "score":0.399
                        },
                    ]
                }
            ],
            "total_num":10
        }
        """
        total_num: int = sum([num for _, num in subset_score_map.values()])
        # weighted_avg_bleu: float = sum([score * num for score, num in subset_score_map.values()]) / total_num
        cate_avg_list = [{'name': subset_name, 'score': score_dict} for subset_name, (score_dict, _) in subset_score_map.items()]
        total_avg_list = defaultdict(float)
        for score_dict, num in subset_score_map.values():
            for metric, score in score_dict.items():
                total_avg_list[metric] += score * num / total_num

        category_d = dict(name="DEFAULT",
                          score=total_avg_list,
                          subset=cate_avg_list)
        
        res_map = dict(name="COCOTest",
                       metric=self.metric_list[0]['name'],
                       score=total_avg_list,
                       category=[category_d],
                       total_num=total_num)
        
        return res_map