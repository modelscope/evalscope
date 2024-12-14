# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Allen Institute, and its affiliates.
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
"""AI2 ARC (Abstraction and Reasoning Corpus) for General Artificial Intelligence Benchmark."""
"""AUTO GENERATED, DO NOT EDIT"""

import datasets
import json
import os

# flake8: noqa

_CITATION = """\
@article{allenai:arc,
      author    = {Peter Clark  and Isaac Cowhey and Oren Etzioni and Tushar Khot and
                    Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
      title     = {Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
      journal   = {arXiv:1803.05457v1},
      year      = {2018},
}
"""

_DESCRIPTION = """\
A new dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in
 advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where the former contains
 only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. We are also
 including a corpus of over 14 million science sentences relevant to the task,
 and an implementation of three neural baseline models for this dataset. We pose ARC as a challenge to the community.

ARC-Easy:
    train: 2251
    test: 2376
    validation: 570

ARC-Challenge:
    train: 1119
    test: 1172
    validation: 299
"""

_URL = 'https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/arc/ARC-V1-Feb2018.zip'

# tasks: ['ARC-Easy', 'ARC-Challenge']


class Ai2ArcConfig(datasets.BuilderConfig):
    """BuilderConfig for Ai2ARC."""

    def __init__(self, **kwargs):
        """BuilderConfig for Ai2Arc.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Ai2ArcConfig, self).__init__(version=datasets.Version('1.0.0', ''), **kwargs)


class Ai2Arc(datasets.GeneratorBasedBuilder):
    """
    The AI2 Reasoning Challenge (ARC) dataset.
    Subset: ARC-Easy, ARC-Challenge.
    """

    VERSION = datasets.Version('1.0.0')
    BUILDER_CONFIGS = [
        Ai2ArcConfig(
            name='ARC-Challenge',
            description="""\
          Challenge Set of 2590 “hard” questions (those that both a retrieval and a co-occurrence method fail to answer correctly)
          """,
        ),
        Ai2ArcConfig(
            name='ARC-Easy',
            description="""\
          Easy Set of 5197 questions
          """,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features({
                'id':
                datasets.Value('string'),
                'question':
                datasets.Value('string'),
                'choices':
                datasets.features.Sequence({
                    'text': datasets.Value('string'),
                    'label': datasets.Value('string')
                }),
                'answerKey':
                datasets.Value('string')
                # These are the features of your dataset like images, labels ...
            }),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage='https://allenai.org/data/arc',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, 'ARC-V1-Feb2018-2')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={'filepath': os.path.join(data_dir, self.config.name, self.config.name + '-Train.jsonl')},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={'filepath': os.path.join(data_dir, self.config.name, self.config.name + '-Test.jsonl')},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={'filepath': os.path.join(data_dir, self.config.name, self.config.name + '-Dev.jsonl')},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding='utf-8') as f:
            for row in f:
                data = json.loads(row)
                answerkey = data['answerKey']
                id_ = data['id']
                question = data['question']['stem']
                choices = data['question']['choices']
                text_choices = [choice['text'] for choice in choices]
                label_choices = [choice['label'] for choice in choices]
                yield id_, {
                    'id': id_,
                    'answerKey': answerkey,
                    'question': question,
                    'choices': {
                        'text': text_choices,
                        'label': label_choices
                    },
                }
