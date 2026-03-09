import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from evalscope.utils.logger import get_logger
from evalscope.utils.url_utils import download_url

logger = get_logger()

# Path to the benchmark metadata directory (individual JSON files per benchmark)
BENCHMARK_META_DIR = Path(__file__).parent.parent / 'benchmarks' / '_meta'


@lru_cache
def check_nltk_data(name: str) -> None:
    """
    Ensure required NLTK resources (such as tokenizers, corpora, and taggers) are available using NLTK's own lookup.
    Uses nltk.data.find to check for resource availability and attempts to download missing resources via NLTK.
    For certain resources (e.g., 'punkt_tab'), falls back to downloading from an external mirror if unavailable from NLTK.
    """  # noqa: E501
    import nltk

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


def _create_empty_benchmark_entry() -> Dict[str, Any]:
    """Create an empty benchmark entry structure."""
    return {
        'meta': {},
        'statistics': {},
        'sample_example': {},
        'readme': {
            'en': '',
            'zh': '',
            'content_hash': '',
            'needs_translation': False,
        },
        'updated_at': '',
    }


def load_benchmark_data(benchmark_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load benchmark metadata from individual JSON files in benchmarks/_meta/.

    Args:
        benchmark_name: Specific benchmark to load, or None to load all.

    Returns:
        Dict keyed by benchmark name. If benchmark_name is specified, returns a
        single-entry dict; otherwise returns all benchmarks.
    """
    if benchmark_name:
        json_path = BENCHMARK_META_DIR / f'{benchmark_name}.json'
        if not json_path.exists():
            return {benchmark_name: _create_empty_benchmark_entry()}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {benchmark_name: data}
        except json.JSONDecodeError:
            return {benchmark_name: _create_empty_benchmark_entry()}
    else:
        result = {}
        if not BENCHMARK_META_DIR.exists():
            return result
        for json_file in BENCHMARK_META_DIR.glob('*.json'):
            name = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    result[name] = json.load(f)
            except json.JSONDecodeError:
                continue
        return result


def save_benchmark_data(data: Dict[str, Any], benchmark_name: Optional[str] = None):
    """
    Save benchmark metadata to individual JSON files in benchmarks/_meta/.

    Args:
        data: Benchmark data. If benchmark_name is None, expects a dict of
              {benchmark_name: benchmark_data}; otherwise expects a single entry.
        benchmark_name: Name of the benchmark to save. If None, saves all entries in data.
    """
    BENCHMARK_META_DIR.mkdir(parents=True, exist_ok=True)

    if benchmark_name:
        json_path = BENCHMARK_META_DIR / f'{benchmark_name}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        for name, benchmark_data in data.items():
            json_path = BENCHMARK_META_DIR / f'{name}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
