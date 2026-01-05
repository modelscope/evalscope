"""
Semantic scoring module that combines NLI, BERTScore, and phonetic similarity.
"""

import jellyfish
import numpy as np
import torch
from bert_score import score as bert_score
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from typing import Any, List, Optional, Tuple

BERT_SCORE_BOUNDS = [-0.1180, 1]
PHONETIC_SCORE_BOUNDS = [0.5, 1]
NLI_SCORE_BOUNDS = [0.0028752246871590614, 0.9661698341369629]


def calculate_bert_score(references: List[str], hypotheses: List[str], device: str) -> torch.Tensor:
    """
    Compute BERTScore F1 for reference-hypothesis pairs.

    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        device: Computing device ('cpu' or 'cuda')

    Returns:
        Tensor of F1 scores
    """
    _, _, f1 = bert_score(
        references,
        hypotheses,
        lang='en',
        rescale_with_baseline=True,
        device=device,
    )
    return f1


def calculate_phonetic_similarity(reference: str, hypothesis: str) -> float:
    """
    Compute phonetic similarity using Soundex + Jaro-Winkler.

    Args:
        reference: Reference text
        hypothesis: Hypothesis text

    Returns:
        Phonetic similarity score between 0 and 1
    """
    ref_soundex = jellyfish.soundex(reference)
    hyp_soundex = jellyfish.soundex(hypothesis)
    return jellyfish.jaro_winkler_similarity(ref_soundex, hyp_soundex)


class SemScorer:
    """
    Semantic Scorer combining NLI, BERTScore, and phonetic similarity.

    The primary contributions to this part were made by Bornali Phukon.
    https://github.com/xiuwenz2/SAPC-template/blob/main/utils/metrics.py
    """

    def __init__(
        self,
        model: str = 'roberta',
        batch_size: int = 32,
        device: Optional[str] = None,
        direction: str = 'avg',
        cross_lingual: bool = False,
        nli_weight: float = 0.4012,
        bert_weight: float = 0.2785,
        phonetic_weight: float = 0.3201,
        **metric_conf
    ):
        """
        Initialize the Semantic Scorer.

        Args:
            model: Model type ('R' for RoBERTa)
            batch_size: Batch size for processing
            device: Computing device ('cpu', 'cuda', or None for auto-detection)
            direction: Direction for NLI scoring ('rh', 'hr', or 'avg')
            cross_lingual: Whether to enable cross-lingual scoring
            nli_weight: Weight for NLI score component
            bert_weight: Weight for BERTScore component
            phonetic_weight: Weight for phonetic similarity component
            metric_conf: Additional metric configuration parameters
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Store configuration
        self.batch_size = batch_size
        self.cross_lingual = cross_lingual
        self.model_type = model
        self.direction = direction
        self.nli_weight = float(nli_weight)
        self.bert_weight = float(bert_weight)
        self.phonetic_weight = float(phonetic_weight)
        self.metric_config = metric_conf

        # Initialize metric placeholders
        self.metric = None
        self.metric_hash = None

        # Initialize NLI model and tokenizer
        self._model, self._tokenizer = self._initialize_model()

    def _initialize_model(self) -> Tuple[torch.nn.Module, Any]:
        """
        Initialize the NLI model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        if self.model_type == 'roberta':
            from transformers import logging

            logging.set_verbosity_error()
            tokenizer = RobertaTokenizer.from_pretrained(
                'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', use_fast=False
            )
            model = RobertaForSequenceClassification.from_pretrained(
                'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
                num_labels=3,
            )
        else:
            raise ValueError(f'Unsupported model type for SemScorer: {self.model_type}')

        model.eval()
        model = model.to(self.device)
        return model, tokenizer

    def collate_input_features(self, premise: str, hypothesis: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare input features for the NLI model.

        Args:
            premise: Premise text
            hypothesis: Hypothesis text

        Returns:
            Tuple of (input_ids, token_type_ids, attention_mask)
        """
        tokenized_input = self._tokenizer.encode_plus(
            premise,
            hypothesis,
            max_length=self._tokenizer.model_max_length,
            return_token_type_ids=True,
            truncation=True
        )

        input_ids = torch.tensor(tokenized_input['input_ids']).long().unsqueeze(0).to(self.device)
        token_type_ids = torch.tensor(tokenized_input['token_type_ids']).long().unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(tokenized_input['attention_mask']).long().unsqueeze(0).to(self.device)

        return input_ids, token_type_ids, attention_mask

    def score_nli(self,
                  refs: List[str],
                  hyps: List[str],
                  direction: Optional[str] = None,
                  formula: str = 'e') -> List[float]:
        """
        Compute NLI scores for reference-hypothesis pairs.

        Args:
            refs: List of reference texts
            hyps: List of hypothesis texts
            direction: Scoring direction ('rh', 'hr', or 'avg')
            formula: Scoring formula ('e' for entailment)

        Returns:
            List of NLI scores
        """
        # Use default direction if none provided
        direction = direction if direction is not None else self.direction

        # Store probabilities for different directions
        probs_rh, probs_hr = {}, {}

        with torch.no_grad():
            # Reference → Hypothesis direction
            if direction in ['rh', 'avg']:
                probs = []
                for ref, hyp in zip(refs, hyps):
                    input_ids, token_type_ids, attention_mask = self.collate_input_features(ref, hyp)
                    logits = self._model(
                        input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None
                    )[0]
                    prob = torch.softmax(logits, 1).detach().cpu().numpy()
                    probs.append(prob)

                concatenated = np.concatenate(probs, 0)
                probs_rh['e'], probs_rh['n'], probs_rh['c'] = concatenated[:, 0], concatenated[:, 1], concatenated[:, 2]

            # Hypothesis → Reference direction
            if direction in ['hr', 'avg']:
                probs = []
                for ref, hyp in zip(refs, hyps):
                    input_ids, token_type_ids, attention_mask = self.collate_input_features(hyp, ref)
                    logits = self._model(
                        input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None
                    )[0]
                    prob = torch.softmax(logits, 1).detach().cpu().numpy()
                    probs.append(prob)

                concatenated = np.concatenate(probs, 0)
                probs_hr['e'], probs_hr['n'], probs_hr['c'] = concatenated[:, 0], concatenated[:, 1], concatenated[:, 2]

            # Select final score based on direction
            if direction == 'rh':
                final_score = probs_rh['e']
            elif direction == 'hr':
                final_score = probs_hr['e']
            elif direction == 'avg':
                final_score = [(s1 + s2) / 2.0 for s1, s2 in zip(probs_rh['e'], probs_hr['e'])]

        return list(final_score)

    def min_max_normalize(self, scores: List[float], thresholds: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max scaling with thresholds.

        Args:
            scores: List of raw scores
            thresholds: List containing [min_threshold, max_threshold]

        Returns:
            List of normalized scores
        """
        assert len(scores) != 0, 'Cannot normalize empty score list'

        min_val, max_val = thresholds
        normalized_scores = [max(min((score - min_val) / (max_val - min_val), 1), 0) for score in scores]
        return normalized_scores

    def score_all(self, refs: List[str], hyps: List[str]) -> List[float]:
        """
        Compute combined semantic scores using weighted combination of
        NLI, BERTScore, and phonetic similarity.

        Args:
            refs: List of reference texts
            hyps: List of hypothesis texts

        Returns:
            List of combined semantic scores
        """
        # Calculate BERTScore and normalize
        bert_scores = calculate_bert_score(refs, hyps, self.device).tolist()
        bert_scores = self.min_max_normalize(bert_scores, BERT_SCORE_BOUNDS)

        # Calculate phonetic similarity scores and normalize
        phonetic_scores = [calculate_phonetic_similarity(ref, hyp) for ref, hyp in zip(refs, hyps)]
        phonetic_scores = self.min_max_normalize(phonetic_scores, PHONETIC_SCORE_BOUNDS)

        # Calculate NLI scores and normalize
        nli_scores = self.score_nli(refs, hyps, formula='e')
        nli_scores = self.min_max_normalize(nli_scores, NLI_SCORE_BOUNDS)

        # Combine scores with weights
        combined_scores = [
            self.nli_weight * nli + self.bert_weight * bert + self.phonetic_weight * phon
            for nli, bert, phon in zip(nli_scores, bert_scores, phonetic_scores)
        ]

        return [float(score) for score in combined_scores]
