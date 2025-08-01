from typing import Any, Dict, List

from evalscope.api.filter import FilterEnsemble
from evalscope.api.registry import get_filter


def build_filter_ensemble(name: str = 'default', filters: Dict[str, Any] = {}) -> FilterEnsemble:
    """
    Create a filtering pipeline.
    """
    filters = []
    for filter_name, filter_args in filters.items():
        filter_cls = get_filter(filter_name)
        if isinstance(filter_args, list):
            filter_function = filter_cls(*filter_args)
        elif isinstance(filter_args, dict):
            filter_function = filter_cls(**filter_args)
        else:
            # Assume single value for simple filters
            filter_function = filter_cls(filter_args)
        # add the filter as a pipeline step
        filters.append(filter_function)

    return FilterEnsemble(name=name, filters=filters)
