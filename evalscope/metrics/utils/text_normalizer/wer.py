import editdistance
import re
import unicodedata
import zhconv

from evalscope.utils.logger import get_logger
from .basic import BasicTextNormalizer
from .chinese import TextNorm as ChineseTextNormalizer
from .english import EnglishTextNormalizer

logger = get_logger()

english_normalizer = EnglishTextNormalizer()
chinese_normalizer = ChineseTextNormalizer(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode='',
)
basic_normalizer = BasicTextNormalizer()

PUNCS = '!,.?;:'


def remove_sp(text, language):
    gt = re.sub(r'<\|.*?\|>', ' ', text)
    gt = re.sub(r'\s+', r' ', gt)  # Replace consecutive spaces in the text with a single space.
    gt = re.sub(f' ?([{PUNCS}])', r'\1', gt)
    gt = gt.lstrip(' ')
    if language == 'cmn_hans':
        gt = re.sub(r'\s+', r'', gt)
    return gt


class EvaluationTokenizer(object):
    """A generic evaluation-time tokenizer, which leverages built-in tokenizers
    in sacreBLEU (https://github.com/mjpost/sacrebleu). It additionally provides
    lowercasing, punctuation removal and character tokenization, which are
    applied after sacreBLEU tokenization.

    Args:
        tokenizer_type (str): the type of sacreBLEU tokenizer to apply.
        lowercase (bool): lowercase the text.
        punctuation_removal (bool): remove punctuation (based on unicode
        category) from text.
        character_tokenization (bool): tokenize the text to characters.
    """

    SPACE = chr(32)
    SPACE_ESCAPE = chr(9601)

    # ALL_TOKENIZER_TYPES = ChoiceEnum(["none", "13a", "intl", "zh", "ja-mecab"])

    def __init__(
        self,
        tokenizer_type: str = '13a',
        lowercase: bool = False,
        punctuation_removal: bool = False,
        character_tokenization: bool = False,
    ):
        from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
        from sacrebleu.tokenizers.tokenizer_char import TokenizerChar
        from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International
        from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
        from sacrebleu.tokenizers.tokenizer_none import NoneTokenizer
        from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

        TOKENIZERS = {
            'none': NoneTokenizer,
            '13a': Tokenizer13a,
            'intl': TokenizerV14International,
            'zh': TokenizerZh,
            'ja-mecab': TokenizerJaMecab,
            'char': TokenizerChar,
        }

        assert tokenizer_type in TOKENIZERS, f'{tokenizer_type}, {TOKENIZERS}'
        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
        self.character_tokenization = character_tokenization
        self.tokenizer = TOKENIZERS[tokenizer_type]()

    @classmethod
    def remove_punctuation(cls, sent: str):
        """Remove punctuation based on Unicode category."""
        return cls.SPACE.join(t for t in sent.split(cls.SPACE) if not all(unicodedata.category(c)[0] == 'P' for c in t))

    def tokenize(self, sent: str):
        tokenized = self.tokenizer(sent)

        if self.punctuation_removal:
            tokenized = self.remove_punctuation(tokenized)

        if self.character_tokenization:
            tokenized = self.SPACE.join(list(tokenized.replace(self.SPACE, self.SPACE_ESCAPE)))

        if self.lowercase:
            tokenized = tokenized.lower()

        return tokenized


def normalize_text(text: str, language: str) -> str:
    """Normalize a single text based on language.
    Includes space/special-token cleanup (remove_sp) and language-specific normalization.
    """
    # cleanup
    text = remove_sp(text, language)

    # language-specific normalization
    if language in ['yue_hant']:
        text = zhconv.convert(text, 'zh-cn')
    if language in ['en']:
        return english_normalizer(text)
    if language in ['cmn_hans']:
        return chinese_normalizer(text)
    return basic_normalizer(text)


def normalize_inputs(ground_truths: list[str], predictions: list[str], language: str) -> tuple[list[str], list[str]]:
    """Normalize batch inputs via normalize_text."""
    refs: list[str] = []
    hyps: list[str] = []
    for gt, pred in zip(ground_truths, predictions):
        refs.append(normalize_text(gt, language))
        hyps.append(normalize_text(pred, language))
    return refs, hyps


def wer(refs: list[str], hyps: list[str], language: str) -> float:
    """Compute WER on already-normalized inputs.
    Inputs must be pre-normalized with normalize_inputs/normalize_text.
    """
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
        tokenizer_type='none',
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )

    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]
        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()
        if language in ['cmn_hans', 'yue_hant']:
            ref_items = [x for x in ''.join(ref_items)]
            pred_items = [x for x in ''.join(pred_items)]
        distance += editdistance.eval(ref_items, pred_items)
        ref_length += len(ref_items)
    return distance / ref_length
