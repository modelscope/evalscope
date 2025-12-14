import nltk
import os
from functools import lru_cache

from evalscope.utils.io_utils import download_url
from evalscope.utils.logger import get_logger

logger = get_logger()


@lru_cache
def check_nltk_data(name: str) -> None:
    """
    Ensure required NLTK tokenizer data is available using NLTK's own lookup.
    Prefer nltk.data.find to check availability rather than filesystem paths.
    Falls back to downloading via NLTK; if punkt_tab is unavailable from NLTK,
    download it from the OSS mirror as a last resort.
    """
    try:

        def has_resource(name: str) -> bool:
            try:
                nltk.data.find(name)
                return True
            except LookupError:
                return False

        if name == 'punkt_tab':
            if not has_resource('tokenizers/punkt_tab'):
                logger.warning('NLTK download for punkt_tab failed, trying mirror.')
                nltk_dir = os.path.join(os.path.expanduser('~'), 'nltk_data/tokenizers')
                os.makedirs(nltk_dir, exist_ok=True)
                punkt_zip = os.path.join(nltk_dir, 'punkt_tab.zip')
                punkt_tab_url = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/open_data/nltk_data/punkt_tab.zip'

                download_url(punkt_tab_url, punkt_zip)
                os.system(f'unzip -o {punkt_zip} -d {nltk_dir}')
        if name == 'stopwords' and not has_resource('corpora/stopwords'):
            nltk.download('stopwords')

        if name == 'averaged_perceptron_tagger_eng' and not has_resource('taggers/averaged_perceptron_tagger'):
            nltk.download('averaged_perceptron_tagger')

    except Exception as e:
        logger.error(f'NLTK data setup failed: {e}')
