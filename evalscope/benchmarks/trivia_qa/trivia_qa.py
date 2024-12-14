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
import json
import os
import pandas as pd

_CITATION = """\
@article{2017arXivtriviaqa,
       author = {{Joshi}, Mandar and {Choi}, Eunsol and {Weld},
                 Daniel and {Zettlemoyer}, Luke},
        title = "{triviaqa: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension}",
      journal = {arXiv e-prints},
         year = 2017,
          eid = {arXiv:1705.03551},
        pages = {arXiv:1705.03551},
archivePrefix = {arXiv},
       eprint = {1705.03551},
}
"""

_DESCRIPTION = """\
TriviaqQA is a reading comprehension dataset containing over 650K question-answer-evidence triples.
"""

_HOMEPAGE = 'https://modelscope.cn/datasets/modelscope/trivia_qa/summary'

_URL = 'https://modelscope.cn/api/v1/datasets/modelscope/trivia_qa/repo?Revision=master&FilePath=trivia_qa.zip'

task_list = ['default']


class TriviaQAConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version('1.0.0'), **kwargs)


class TriviaQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [TriviaQAConfig(name=task_name, ) for task_name in task_list]

    def _info(self):
        features = datasets.Features({
            'input': [{
                'role': datasets.features.Value('string'),
                'content': datasets.features.Value('string'),
            }],
            'ideal': [datasets.Value('string')],
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': os.path.join(data_dir, 'trivia_qa/test.jsonl'),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('dev'),
                gen_kwargs={
                    'filepath': os.path.join(data_dir, 'trivia_qa/dev.jsonl'),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            contents = [json.loads(line) for line in f.readlines()]
            for i, instance in enumerate(contents):
                yield i, instance
