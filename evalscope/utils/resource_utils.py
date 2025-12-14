import nltk
import os
import zipfile
from functools import lru_cache

from evalscope.utils.io_utils import download_url
from evalscope.utils.logger import get_logger

logger = get_logger()


@lru_cache
def check_nltk_data(name: str) -> None:
    """
    Ensure required NLTK resources (such as tokenizers, corpora, and taggers) are available using NLTK's own lookup.
    Uses nltk.data.find to check for resource availability and attempts to download missing resources via NLTK.
    For certain resources (e.g., 'punkt_tab'), falls back to downloading from an external mirror if unavailable from NLTK.
    """  # noqa: E501

    def has_resource(name: str) -> bool:
        try:
            nltk.data.find(name)
            return True
        except LookupError:
            return False

    try:
        if name == 'punkt_tab':
            # Punkt tab may not be bundled; check tokenizers path
            if not has_resource('tokenizers/punkt_tab'):
                logger.warning('NLTK download for punkt_tab failed, trying mirror.')
                nltk_dir = os.path.join(os.path.expanduser('~'), 'nltk_data/tokenizers')
                os.makedirs(nltk_dir, exist_ok=True)
                punkt_zip = os.path.join(nltk_dir, 'punkt_tab.zip')
                punkt_tab_url = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/open_data/nltk_data/punkt_tab.zip'

                download_url(punkt_tab_url, punkt_zip)

                try:
                    with zipfile.ZipFile(punkt_zip, 'r') as zip_ref:
                        zip_ref.extractall(nltk_dir)
                finally:
                    if os.path.exists(punkt_zip):
                        os.remove(punkt_zip)
        if name == 'stopwords' and not has_resource('corpora/stopwords'):
            nltk.download('stopwords')

        if name == 'averaged_perceptron_tagger_eng' and not has_resource('taggers/averaged_perceptron_tagger'):
            nltk.download('averaged_perceptron_tagger')
    except Exception as e:
        logger.error(f'NLTK data setup failed: {e}')
