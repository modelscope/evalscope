import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from evalscope.config import TaskConfig

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
    For certain resources, falls back to downloading from external mirrors (ModelScope OSS or Gitee) if unavailable.
    """  # noqa: E501
    import nltk

    GITEE_MIRROR = 'https://gitee.com/yzy0612/nltk_data/raw/gh-pages/packages'

    MIRROR_MAP = {
        'punkt_tab': {
            'resource_path':
            'tokenizers/punkt_tab',
            'zip_name':
            'punkt_tab.zip',
            'extract_dir':
            'tokenizers',
            'urls': [
                'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/open_data/nltk_data/punkt_tab.zip',
                f'{GITEE_MIRROR}/tokenizers/punkt_tab.zip',
            ],
        },
        'stopwords': {
            'resource_path': 'corpora/stopwords',
            'zip_name': 'stopwords.zip',
            'extract_dir': 'corpora',
            'urls': [
                f'{GITEE_MIRROR}/corpora/stopwords.zip',
            ],
        },
        'averaged_perceptron_tagger_eng': {
            'resource_path': 'taggers/averaged_perceptron_tagger',
            'zip_name': 'averaged_perceptron_tagger.zip',
            'extract_dir': 'taggers',
            'urls': [
                f'{GITEE_MIRROR}/taggers/averaged_perceptron_tagger.zip',
            ],
        },
    }

    def has_resource(resource_path: str) -> bool:
        try:
            nltk.data.find(resource_path)
            return True
        except LookupError:
            return False

    def download_from_mirrors(meta: dict) -> None:
        nltk_base = os.path.expanduser('~/nltk_data')
        extract_dir = os.path.join(nltk_base, meta['extract_dir'])
        os.makedirs(extract_dir, exist_ok=True)
        zip_path = os.path.join(extract_dir, meta['zip_name'])
        last_error = None
        for url in meta['urls']:
            try:
                download_url(url, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                return
            except Exception as e:
                last_error = e
                logger.warning(f'Failed to download from {url}: {e}')
            finally:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
        raise RuntimeError(f'All mirrors failed for "{meta["zip_name"]}": {last_error}')

    try:
        if name in MIRROR_MAP:
            meta = MIRROR_MAP[name]
            if not has_resource(meta['resource_path']):
                logger.warning(f'NLTK resource "{name}" not found, downloading from mirror.')
                download_from_mirrors(meta)
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


def compute_eval_total_count(task_config: 'TaskConfig') -> Optional[int]:
    """Estimate the total number of evaluation samples for a task configuration.

    Reads per-subset ``sample_count`` values from the bundled ``_meta`` JSON
    files and applies ``limit`` (per-subset cap) and ``repeats`` (multiplier).
    Returns ``None`` if the count cannot be determined (e.g. missing meta file
    or unknown dataset).

    Calculation per dataset::

        effective = min(sample_count, limit)  # apply limit first
        contribution = effective * repeats    # then multiply by repeats

    """
    total = 0

    for dataset_name in task_config.datasets:
        entry = load_benchmark_data(dataset_name).get(dataset_name, {})
        if not entry.get('statistics'):
            logger.debug(f'No meta file found for dataset "{dataset_name}", skipping total_count estimate.')
            return None

        subset_stats: List[dict] = entry.get('statistics', {}).get('subset_stats', [])
        if not subset_stats:
            logger.debug(f'No subset_stats in meta for "{dataset_name}", skipping total_count estimate.')
            return None

        subset_count_map = {s['name']: s['sample_count'] for s in subset_stats}

        # Determine which subsets are active (user override or full list)
        dataset_args = task_config.dataset_args.get(dataset_name, {})
        active_subsets = dataset_args.get('subset_list', None)
        if active_subsets is None:
            active_subsets = entry.get('meta', {}).get('subset_list', list(subset_count_map.keys()))

        limit = task_config.limit
        repeats = task_config.repeats

        for subset in active_subsets:
            sample_count = subset_count_map.get(subset)
            if sample_count is None:
                continue
            # Apply limit per subset first
            if limit is not None:
                if isinstance(limit, float):
                    effective = int(sample_count * limit)
                else:
                    effective = min(sample_count, int(limit))
            else:
                effective = sample_count
            # Then multiply by repeats
            total += effective * repeats

    return total if total > 0 else None
