import torch
from typing import List

from evalscope.utils.import_utils import check_import


def compute_bertscore_one_sample(
    predictions: List[str], references: List[str], lang: str = 'de', model_type: str = 'xlm-roberta-large'
) -> dict:
    """Compute BERTScore for a single prediction-reference pair.

    Args:
        predictions: List containing one predicted translation.
        references: List containing one reference translation.
        lang: Target language code (default: "de").
        model_type: Pretrained model used for embedding (default: "xlm-roberta-large").

    Returns:
        A dictionary containing BERTScore precision, recall, and F1.
    """
    check_import('bert_score', 'bert_score', raise_error=True, feature_name='Text semantic similarity metrics')
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions, references, lang=lang, model_type=model_type, rescale_with_baseline=True, verbose=False
    )
    return {
        'bertscore-precision': round(P[0].item(), 6),
        'bertscore-recall': round(R[0].item(), 6),
        'bertscore-f1': round(F1[0].item(), 6),
    }


def compute_bertscore_batch(
    predictions: List[str],
    references: List[str],
    lang: str = 'de',
    model_type: str = 'xlm-roberta-large',
    batch_size: int = 64,
) -> List[dict]:
    """Compute BERTScore for a batch of predictions and references.

    Args:
        predictions: List of model predictions.
        references: List of reference translations.
        lang: Target language code (default: "de").
        model_type: Pretrained model for BERTScore (default: "xlm-roberta-large").
        batch_size: Number of samples per batch (default: 64).

    Returns:
        A list of dictionaries with precision, recall, and F1 for each sample.
    """
    check_import('bert_score', 'bert_score', raise_error=True, feature_name='Text semantic similarity metrics')
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        rescale_with_baseline=True,
        verbose=False
    )
    results = []
    for p, r, f in zip(P, R, F1):
        results.append({
            'bertscore-precision': round(p.item(), 6),
            'bertscore-recall': round(r.item(), 6),
            'bertscore-f1': round(f.item(), 6),
        })
    return results


def get_comet_model():
    """Lazily load and cache the COMET model.

    Automatically downloads and loads the `Unbabel/wmt22-comet-da` model.
    Moves the model to GPU if available.

    Returns:
        A COMET model instance.
    """
    check_import('comet', 'comet', raise_error=True, feature_name='Text translation metrics')
    from comet import download_model, load_from_checkpoint

    model_path = download_model('Unbabel/wmt22-comet-da')
    model = load_from_checkpoint(model_path)
    if torch.cuda.is_available():
        model.cuda()
    return model


def compute_comet_score_one_sample(src: str, mt: str, ref: str) -> dict:
    """Compute COMET score for a single translation sample.

    Args:
        src: Source sentence.
        mt: Machine-translated sentence.
        ref: Human reference translation.

    Returns:
        A dictionary containing the COMET score.
    """
    model = get_comet_model()
    data = [{'src': src, 'mt': mt, 'ref': ref}]
    output = model.predict(data, batch_size=10, gpus=1 if torch.cuda.is_available() else 0)
    return {'comet': round(output['scores'][0], 6)}


def compute_comet_batch(src_list: List[str],
                        mt_list: List[str],
                        ref_list: List[str],
                        model=None,
                        batch_size: int = 32) -> List[dict]:
    """Compute COMET scores in batch mode, with optional model reuse.

    Args:
        src_list: List of source sentences.
        mt_list: List of model translations.
        ref_list: List of reference translations.
        model: Preloaded COMET model for reuse (optional).
        batch_size: Batch size for prediction (default: 32).

    Returns:
        A list of dictionaries containing COMET scores for each sample.
    """
    check_import('comet', 'comet', raise_error=True, feature_name='Text translation metrics')
    from comet import download_model, load_from_checkpoint

    if model is None:
        model_path = download_model('Unbabel/wmt22-comet-da')
        model = load_from_checkpoint(model_path)
        if torch.cuda.is_available():
            model.cuda()

    data = [{'src': s, 'mt': m, 'ref': r} for s, m, r in zip(src_list, mt_list, ref_list)]
    output = model.predict(data, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0)
    scores = output['scores']
    return [{'comet': round(score, 6)} for score in scores]


def compute_meteor_score_one_sample(pred: str, ref: str) -> dict:
    """Compute METEOR score for a single translation pair.

    Args:
        pred: Predicted translation.
        ref: Reference translation.

    Returns:
        A dictionary containing the METEOR score.
    """
    check_import('nltk', 'nltk', raise_error=True, feature_name='Text translation metrics')
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    ref_tokens = word_tokenize(ref)
    pred_tokens = word_tokenize(pred)
    output = meteor_score([ref_tokens], pred_tokens)
    return {'meteor': output}


def compute_bleu_batch(preds: List[str], refs: List[str]) -> List[dict]:
    """Compute BLEU scores for a batch of predictions and references.

    Args:
        preds: List of predicted translations.
        refs: List of reference translations.

    Returns:
        A list of BLEU score dictionaries for each sample.
    """
    from evalscope.metrics import bleu_ngram_one_sample
    results = []
    for p, r in zip(preds, refs):
        results.append(bleu_ngram_one_sample(p, r))
    return results
