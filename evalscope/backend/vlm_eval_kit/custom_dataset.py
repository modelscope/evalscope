import numpy as np
import os
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import d2df, dump, load


class CustomDataset:

    def load_data(self, dataset):
        # customize the loading of the dataset
        data_path = os.path.join(os.path.expanduser('~/LMUData'), f'{dataset}.tsv')
        return load(data_path)

    def build_prompt(self, line):
        msgs = ImageBaseDataset.build_prompt(self, line)
        # add a hint or custom instruction here
        msgs[-1]['value'] += '\nAnswer the question using a single word or phrase.'
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x).lower() for x in data['answer']]

        print(data)

        # ========compute the evaluation metrics as you need =========
        # exact match
        result = np.mean(data['answer'] == data['prediction'])
        ret = {'Overall': result}
        ret = d2df(ret).round(2)

        # save the result
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret
        # ============================================================


# override the default dataset class
CustomVQADataset.load_data = CustomDataset.load_data
CustomVQADataset.build_prompt = CustomDataset.build_prompt
CustomVQADataset.evaluate = CustomDataset.evaluate
