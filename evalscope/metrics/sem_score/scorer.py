#!/usr/bin/env python3
# Author: xiuwenz2@illinois.edu
# Date: Oct. 08, 2024

import jellyfish
import numpy as np
import torch
from bert_score import score as bert_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def calculate_bert_score(references, hypotheses, device):
    """
    Compute BERTScore F1 for reference-hypothesis pairs.
    """
    _, _, f1 = bert_score(
        references,
        hypotheses,
        lang='en',
        rescale_with_baseline=True,
        device=device,
    )
    return f1


def calculate_phonetic_similarity(reference, hypothesis):
    """
    Compute phonetic similarity using Soundex + Jaro-Winkler.
    """
    ref_soundex = jellyfish.soundex(reference)
    hyp_soundex = jellyfish.soundex(hypothesis)
    return jellyfish.jaro_winkler_similarity(ref_soundex, hyp_soundex)


# =========================
# Semantic Scoring Class
# =========================


class SemScorer:
    """
    Combined semantic score using:
      - NLI entailment probability
      - BERTScore
      - Phonetic similarity
    """

    def __init__(
        self,
        model_type='roberta',
        batch_size=32,
        device=None,
        direction='avg',
        cross_lingual=False,
        nli_weight=0.4012,
        bert_weight=0.2785,
        phonetic_weight=0.3201,
        model_id_or_path='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
        **metric_conf,
    ):
        self.model_id_or_path = model_id_or_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.cross_lingual = cross_lingual
        self.model_type = model_type
        self.direction = direction

        self.nli_weight = float(nli_weight)
        self.bert_weight = float(bert_weight)
        self.phonetic_weight = float(phonetic_weight)

        self.metric_config = metric_conf
        self.metric = None
        self.metric_hash = None

        self.model, self.tokenizer = self._load_nli_model()

    # ---------------------
    # Model Utilities
    # ---------------------

    def _load_nli_model(self):
        """
        Load pretrained NLI model and tokenizer.
        """
        if self.model_type == 'roberta':
            model_name = self.model_id_or_path
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=self.cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=3, cache_dir=self.cache_dir
            )

        model.eval()
        model.to(self.device)
        return model, tokenizer

    def _collate_inputs(self, premise, hypothesis):
        """
        Tokenize and prepare model inputs.
        """
        encoded = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            max_length=self.tokenizer.model_max_length,
            return_token_type_ids=True,
            truncation=True,
        )

        input_ids = torch.tensor(encoded['input_ids']).long().unsqueeze(0).to(self.device)
        token_type_ids = (torch.tensor(encoded['token_type_ids']).long().unsqueeze(0).to(self.device))
        attention_mask = (torch.tensor(encoded['attention_mask']).long().unsqueeze(0).to(self.device))

        return input_ids, token_type_ids, attention_mask

    # ---------------------
    # NLI Scoring
    # ---------------------

    def score_nli(self, references, hypotheses, direction=None, formula='e'):
        """
        Compute entailment probability using NLI.
        """
        direction = direction or self.direction
        probs_rh, probs_hr = {}, {}

        with torch.no_grad():
            if direction in {'rh', 'avg'}:
                probs = []
                for ref, hyp in zip(references, hypotheses):
                    inputs = self._collate_inputs(ref, hyp)
                    logits = self.model(*inputs)[0]
                    probs.append(torch.softmax(logits, dim=1).cpu().numpy())

                concat = np.concatenate(probs, axis=0)
                probs_rh['e'] = concat[:, 0]

            if direction in {'hr', 'avg'}:
                probs = []
                for ref, hyp in zip(references, hypotheses):
                    inputs = self._collate_inputs(hyp, ref)
                    logits = self.model(*inputs)[0]
                    probs.append(torch.softmax(logits, dim=1).cpu().numpy())

                concat = np.concatenate(probs, axis=0)
                probs_hr['e'] = concat[:, 0]

        if direction == 'rh':
            scores = probs_rh['e']
        elif direction == 'hr':
            scores = probs_hr['e']
        else:
            scores = [(a + b) / 2 for a, b in zip(probs_rh['e'], probs_hr['e'])]

        return list(scores)

    # ---------------------
    # Final Combined Score
    # ---------------------

    def score_all(self, hypotheses, references):
        """
        Compute combined semantic score for reference-hypothesis pairs.
        """
        # BERTScore
        bert_scores = calculate_bert_score(references, hypotheses, self.device).tolist()
        BERT_SCORE_BOUNDS = [-0.1180, 1]
        bert_scores = self._min_max_normalize(bert_scores, BERT_SCORE_BOUNDS)

        # Phonetic similarity
        phonetic_scores = [calculate_phonetic_similarity(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        PHONETIC_SCORE_BOUNDS = [0.5, 1]
        phonetic_scores = self._min_max_normalize(phonetic_scores, PHONETIC_SCORE_BOUNDS)

        # NLI
        nli_scores = self.score_nli(references, hypotheses)
        NLI_SCORE_BOUNDS = [0.0028752246871590614, 0.9661698341369629]
        nli_scores = self._min_max_normalize(nli_scores, NLI_SCORE_BOUNDS)

        # Weighted sum
        combined_scores = [
            self.nli_weight * nli + self.bert_weight * bert + self.phonetic_weight * phon
            for nli, bert, phon in zip(nli_scores, bert_scores, phonetic_scores)
        ]

        return combined_scores

    # ---------------------
    # Helpers
    # ---------------------

    @staticmethod
    def _min_max_normalize(scores, bounds):
        """
        Min-max normalize scores into [0, 1].
        """
        low, high = bounds
        return [max(min((s - low) / (high - low), 1.0), 0.0) for s in scores]
