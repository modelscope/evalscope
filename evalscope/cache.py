# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Union

import cachetools
from cachetools import Cache as CachetoolsCache
from pympler import asizeof
from datetime import datetime, timedelta
import pickle

from evalscope.constants import DEFAULT_ROOT_CACHE_DIR
from evalscope.utils.logger import get_logger

logger = get_logger()


DEFAULT_CACHE_MAXSIZE = 1 * 1024 * 1024 * 1024  # 1 GB
DEFAULT_CACHE_EXPIRE = 60 * 60 * 24             # 1 day (seconds)
DEFAULT_MEM_CACHE_PATH = os.environ.get('MEM_CACHE_PATH',
                                        os.path.join(os.path.expanduser(DEFAULT_ROOT_CACHE_DIR),
                                                     'mem_cache', 'global_cache.pkl'))


class Cache:

    # TODO:   by xingjun.wxj@alibaba-inc.com
    #  1. atomic operation for saving cache
    #  2. consider the distributed env

    @classmethod
    def lru_cache(cls, maxsize: int = DEFAULT_CACHE_MAXSIZE):
        return cachetools.LRUCache(maxsize=maxsize, getsizeof=asizeof.asizeof)

    @classmethod
    def ttl_cache(cls, max_size: float = DEFAULT_CACHE_MAXSIZE, expire: float = DEFAULT_CACHE_EXPIRE):
        return cachetools.TTLCache(maxsize=max_size,
                                   ttl=timedelta(seconds=expire),
                                   timer=datetime.now,
                                   getsizeof=asizeof.asizeof)

    @classmethod
    def load(cls, path: str) -> Union[CachetoolsCache, None]:
        """
        Load cache from disk. Pickle is used for serialization.

        Args:
            path: The local path to load the cache.

        Returns:
            The cache instance loaded from disk. Should be cachetools.Cache or None.
        """
        if os.path.exists(path):
            logger.info(f'** Loading cache from {path} ...')
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            return None

    @classmethod
    def save(cls, cache: CachetoolsCache, path: str = DEFAULT_MEM_CACHE_PATH):
        """
        Dump memory cache to disk. Pickle is used for serialization.

        Args:
            cache: The cache instance to be saved.
            path: The local path to save the cache.

        Returns: None
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f'** Cache saved to {path} !')


def init_mem_cache(method: str = 'ttl', cache_file_path: str = DEFAULT_MEM_CACHE_PATH) -> CachetoolsCache:
    """
    Initialize memory cache.

    Args:
        method (str): 'ttl' or 'lru', see https://cachetools.readthedocs.io/en/latest/ for details.
        cache_file_path (str): The local cache path. Should be a pickle file.

    Returns:
        The cache instance. Should be cachetools.Cache.
    """
    logger.info(f'** Initializing memory cache with method `{method}` ... \n')
    mem_cache = Cache.load(path=cache_file_path)
    if mem_cache is None:
        if method == 'ttl':
            mem_cache = Cache.ttl_cache()
        elif method == 'lru':
            mem_cache = Cache.lru_cache()
        else:
            raise ValueError(f'Unknown cache method {method}. Please choose from `ttl` or `lru`.')

    return mem_cache
