import contextlib
import io
import shutil
import string
from collections import Counter
from typing import Dict, List, Tuple

from evalscope.api.metric import AggScore, SampleScore
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

CAPTION_METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
CAPTION_MAIN_SCORE = 'CIDEr'


def compute_caption_scores(predictions: List[str], references: List[List[str]]) -> List[Dict[str, float]]:
    """Compute COCO-style caption metrics for a batch of predictions."""
    return _caption_scores(predictions, references)[1]


def aggregate_caption_scores(sample_scores: List[SampleScore]) -> List[AggScore]:
    """Score all captions as one corpus, as COCOEvalCap does.

    Batch scoring runs in eval_batch_size windows, but CIDEr takes its IDF from the captions passed to
    one call (a lone caption gets IDF 0, so CIDEr 0) and BLEU pools n-gram counts over the corpus, so
    neither equals an average of per-window sample scores.
    """
    scored = [sample_score for sample_score in sample_scores if sample_score.score.value]
    if not scored:
        return []
    corpus, _ = _caption_scores(
        [sample_score.score.extracted_prediction or '' for sample_score in scored],
        [sample_score.sample_metadata['references'] for sample_score in scored],
    )
    ids = [sample_score.sample_id for sample_score in scored]
    return [
        AggScore(score=corpus[metric], metric_name=metric, aggregation='mean', num=len(scored), ids=ids)
        for metric in CAPTION_METRICS
    ]


def _caption_scores(
    predictions: List[str], references: List[List[str]]
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    check_import(
        module_name='pycocoevalcap',
        extra='caption',
        feature_name='caption benchmark metrics',
        raise_error=True,
    )
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    gts = {}
    res = {}
    for index, (prediction, sample_references) in enumerate(zip(predictions, references)):
        gts[index] = [{'caption': reference} for reference in sample_references]
        res[index] = [{'caption': prediction}]

    use_official_java = shutil.which('java') is not None
    if use_official_java:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
    else:
        logger.warning(
            'Java is not available; caption metrics will use a pure-Python tokenizer and METEOR fallback. '
            'Install Java to reproduce the official COCO caption tokenizer and METEOR scores.'
        )
        gts = _simple_caption_tokenize(gts)
        res = _simple_caption_tokenize(res)

    corpus = {}
    results = [{metric: 0.0 for metric in CAPTION_METRICS} for _ in predictions]
    with contextlib.redirect_stdout(io.StringIO()):
        bleu_corpus, bleu_scores = Bleu(4).compute_score(gts, res)
    for metric_index, metric_name in enumerate(['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']):
        corpus[metric_name] = float(bleu_corpus[metric_index])
        for sample_index, sample_score in enumerate(bleu_scores[metric_index]):
            results[sample_index][metric_name] = float(sample_score)

    for metric_name, scorer in [('ROUGE_L', Rouge()), ('CIDEr', Cider())]:
        with contextlib.redirect_stdout(io.StringIO()):
            corpus_score, sample_scores = scorer.compute_score(gts, res)
        corpus[metric_name] = float(corpus_score)
        for sample_index, sample_score in enumerate(sample_scores):
            results[sample_index][metric_name] = float(sample_score)

    if use_official_java:
        from pycocoevalcap.meteor.meteor import Meteor

        with contextlib.redirect_stdout(io.StringIO()):
            corpus_score, sample_scores = Meteor().compute_score(gts, res)
        corpus['METEOR'] = float(corpus_score)
        for sample_index, sample_score in enumerate(sample_scores):
            results[sample_index]['METEOR'] = float(sample_score)
    else:
        for sample_index, (prediction, sample_references) in enumerate(zip(predictions, references)):
            results[sample_index]['METEOR'] = _simple_meteor_score(prediction, sample_references)
        corpus['METEOR'] = sum(result['METEOR'] for result in results) / len(results)

    return corpus, results


def _caption_tokens(text: str) -> List[str]:
    punctuation = string.punctuation.replace(':', '')
    text = text.lower()
    text = ''.join(' ' if char in punctuation else char for char in text)
    return [token for token in text.split() if token]


def _simple_caption_tokenize(data: Dict[int, List[Dict[str, str]]]) -> Dict[int, List[str]]:
    tokenized = {}
    for image_id, annotations in data.items():
        tokenized[image_id] = [' '.join(_caption_tokens(annotation['caption'])) for annotation in annotations]
    return tokenized


def _simple_meteor_score(prediction: str, references: List[str]) -> float:
    prediction_tokens = _caption_tokens(prediction)
    if not prediction_tokens:
        return 0.0

    best_score = 0.0
    prediction_counts = Counter(prediction_tokens)
    for reference in references:
        reference_tokens = _caption_tokens(reference)
        if not reference_tokens:
            continue
        reference_counts = Counter(reference_tokens)
        matches = sum(min(count, reference_counts[token]) for token, count in prediction_counts.items())
        if matches == 0:
            continue
        precision = matches / len(prediction_tokens)
        recall = matches / len(reference_tokens)
        score = (10 * precision * recall) / (recall + 9 * precision)
        best_score = max(best_score, score)
    return best_score
