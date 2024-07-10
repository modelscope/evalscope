# coding=utf-8
"""COCO2014 Data Set"""
import csv
import datasets

# Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{'{a} }r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  archivePrefix = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

# Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
COCO is a large-scale object detection, segmentation, and captioning dataset.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = "http://cocodataset.org/#home"

# Add the licence for the dataset here if you can find it
_LICENSE = "cc-by-4.0"

_URLS = {
    "train": "http://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/coco/2014/train.csv.zip",
    "valid": "http://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/coco/2014/val.csv.zip"
}


class COCO2014Caption(datasets.GeneratorBasedBuilder):
    """COCO2014Caption Data Set"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="coco_2014_caption",
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
        )
    ]
    features = {
                "uniq_id": datasets.Value("string"),
                "image_id": datasets.Value("string"),
                "caption": datasets.Value("string"),
                "image": datasets.Image(),
            }

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(self.features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": dl_manager.iter_files([data_files["train"]]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "files": dl_manager.iter_files([data_files["valid"]]),
                },
            ),
        ]

    def _generate_examples(self, files):
        """This function returns the examples in the raw form."""
        idx = 0
        for file_name in files:
            with open(file_name, encoding="utf8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    example = {feat: row[feat] for feat in self.features.keys()}
                    yield idx, example
                    idx += 1