# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .combinator import (
        gen_table,
        get_data_frame,
        get_report_list,
        percentage_weighted_average_from_subsets,
        unweighted_average_from_subsets,
        weighted_average_from_subsets,
    )
    from .generator import ReportGenerator
    from .report import Category, Metric, Report, ReportKey, Subset

else:
    _import_structure = {
        'combinator': [
            'gen_table',
            'get_data_frame',
            'get_report_list',
            'weighted_average_from_subsets',
            'unweighted_average_from_subsets',
            'percentage_weighted_average_from_subsets',
        ],
        'generator': [
            'ReportGenerator',
        ],
        'report': [
            'Category',
            'Report',
            'ReportKey',
            'Subset',
            'Metric',
        ],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
