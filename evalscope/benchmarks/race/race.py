# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datasets
import os
import pandas as pd

_CITATION = """\
@inproceedings{lai-etal-2017-race,
    title = "{RACE}: Large-scale {R}e{A}ding Comprehension Dataset From Examinations",
    author = "Lai, Guokun  and
      Xie, Qizhe  and
      Liu, Hanxiao  and
      Yang, Yiming  and
      Hovy, Eduard",
    booktitle = "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D17-1082",
    doi = "10.18653/v1/D17-1082",
    pages = "785--794",
}
"""

_DESCRIPTION = """\
RACE is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions.
"""

_HOMEPAGE = 'https://modelscope.cn/datasets/modelscope/race/summary'

_URL = 'https://modelscope.cn/api/v1/datasets/modelscope/race/repo?Revision=master&FilePath=race.zip'

task_list = [
    'high',
    'middle',
]


class RACEConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version('1.0.0'), **kwargs)


class RACE(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [RACEConfig(name=task_name, ) for task_name in task_list]

    def _info(self):
        features = datasets.Features({
            'example_id': datasets.Value('string'),
            'article': datasets.Value('string'),
            'answer': datasets.Value('string'),
            'question': datasets.Value('string'),
            'options': [datasets.Value('string')],
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': os.path.join(data_dir, f'race/test/{task_name}-00000-of-00001.parquet'),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'filepath': os.path.join(data_dir, f'race/val/{task_name}-00000-of-00001.parquet'),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': os.path.join(data_dir, f'race/train/{task_name}-00000-of-00001.parquet'),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_parquet(filepath)
        df.columns = ['example_id', 'article', 'answer', 'question', 'options']

        for i, instance in enumerate(df.to_dict(orient='records')):
            yield i, instance
