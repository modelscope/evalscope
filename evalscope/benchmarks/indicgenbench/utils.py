# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared loading, prompting and scoring for the IndicGenBench adapters (Singh et al., ACL 2024)."""

import json
from typing import Any, Dict, List, Optional, Type, Union

from evalscope.api.benchmark import DefaultDataAdapter
from evalscope.api.dataset import DataLoader, Dataset, Sample, build_dataset_from_records, download_dataset_file
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score

PAPER_URL = 'https://aclanthology.org/2024.acl-long.595/'

# The 29 Indic languages of IndicGenBench, with the English names of the paper (Table 7).
LANGUAGE_NAMES = {
    'as': 'Assamese',
    'awa': 'Awadhi',
    'bgc': 'Haryanvi',
    'bho': 'Bhojpuri',
    'bn': 'Bengali',
    'bo': 'Tibetan',
    'brx': 'Bodo',
    'gbm': 'Garhwali',
    'gom': 'Konkani',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'hne': 'Chhattisgarhi',
    'hoj': 'Rajasthani',
    'kn': 'Kannada',
    'mai': 'Maithili',
    'ml': 'Malayalam',
    'mni': 'Manipuri',
    'mr': 'Marathi',
    'mup': 'Malvi',
    'mwr': 'Marwari',
    'ne': 'Nepali',
    'or': 'Odia',
    'pa': 'Punjabi',
    'ps': 'Pashto',
    'sa': 'Sanskrit',
    'sat': 'Santali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ur': 'Urdu',
}
LANGUAGES = list(LANGUAGE_NAMES)


def language_name(code: str) -> str:
    return 'English' if code == 'en' else LANGUAGE_NAMES[code]


# chrF as defined by SacreBLEU's defaults: character n-grams up to 6, no word n-grams (chrF, not chrF++), beta 2.
CHRF_SETTINGS = {'char_order': 6, 'word_order': 0, 'beta': 2}


def first_line(text: str) -> str:
    """The first non-empty line of a completion: each prompt asks for a single answer after its cue."""
    return text.strip().split('\n', 1)[0].strip()


def corpus_chrf(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level chrF in [0, 1], computed by SacreBLEU from character n-gram counts pooled over all segments."""
    from sacrebleu.metrics import CHRF

    return CHRF(**CHRF_SETTINGS).corpus_score(hypotheses, [references]).score / 100.0


def sentence_chrf(hypothesis: str, reference: str) -> float:
    """Segment-level chrF in [0, 1], recorded per sample for inspection; subsets report `corpus_chrf`."""
    from sacrebleu.metrics import CHRF

    return CHRF(**CHRF_SETTINGS).sentence_score(hypothesis, [reference]).score / 100.0


class IndicGenBenchAdapter(DefaultDataAdapter):
    """Read released JSON files and score the paper's generation tasks with corpus chrF.

    Every IndicGenBench file keeps its rows under a top-level `examples` key, next to a canary string, so the files
    are read directly. An n-shot example uses the same template as the test item followed by its reference, and
    examples are separated by a blank line (Table 15 of the paper).
    """

    def subset_file(self, subset: str, split: str) -> str:
        """Name of the released file holding `split` of `subset`."""
        raise NotImplementedError

    def check_record(self, subset: str, record: Dict[str, Any]) -> Optional[str]:
        """Describe why `record` does not belong to `subset`, or return None when it does."""
        return None

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        return self._build_subset(subset, self.eval_split, self.limit, self.repeats, self.shuffle)

    def load_fewshot_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        return self._build_subset(subset, self.train_split, self.few_shot_num, 1, self.few_shot_random)

    def _build_subset(
        self, subset: str, split: str, limit: Optional[Union[int, float]], repeats: int, shuffle: bool
    ) -> Dataset:
        return build_dataset_from_records(
            records=self._read_records(subset, split),
            sample_fields=self.record_to_sample,
            name=subset,
            location=self.dataset_id,
            limit=limit,
            repeats=repeats,
            shuffle=shuffle,
            seed=self.seed,
            filter_func=self.sample_filter,
        )

    def _read_records(self, subset: str, split: str) -> List[Dict[str, Any]]:
        file_name = self.subset_file(subset, split)
        path = download_dataset_file(
            data_id_or_path=self.dataset_id,
            file_path=file_name,
            data_source=self.dataset_hub,
            revision=self.dataset_revision,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        )
        with open(path, encoding='utf-8') as f:
            records = json.load(f)['examples']
        for record in records:
            problem = self.check_record(subset, record)
            if problem:
                raise ValueError(f'{file_name}: {problem}')
        return records

    def sample_to_fewshot(self, sample: Sample) -> str:
        return f'{self.format_prompt_template(sample)} {sample.target}'

    def format_fewshot_template(self, fewshot: str, sample: Sample) -> str:
        return f'{fewshot}\n\n{self.format_prompt_template(sample)}'

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return first_line(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        return Score(
            value={'chrf': sentence_chrf(filtered_prediction, reference)},
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
            main_score_name='chrf',
            metadata={'reference': reference},
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Corpus-level chrF of the subset, from the scored output and reference of every sample."""
        scored = [ss for ss in sample_scores if 'chrf' in ss.score.value]
        if not scored:
            return []
        hypotheses = [ss.score.extracted_prediction or '' for ss in scored]
        references = [ss.score.metadata['reference'] for ss in scored]
        return [
            AggScore(
                score=corpus_chrf(hypotheses, references),
                metric_name='chrf',
                aggregation='official',
                num=len(scored),
                ids=[ss.sample_id for ss in scored],
                metadata={'type': 'sacrebleu-corpus-chrf', **CHRF_SETTINGS},
            )
        ]
