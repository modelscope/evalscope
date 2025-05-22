# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .combinator import gen_report_table, gen_table, get_data_frame, get_report_list
    from .generator import ReportGenerator
    from .utils import Category, Report, ReportKey, Subset

else:
    _import_structure = {
        'combinator': [
            'gen_table',
            'get_data_frame',
            'get_report_list',
            'gen_report_table',
        ],
        'generator': [
            'ReportGenerator',
        ],
        'utils': [
            'Category',
            'Report',
            'ReportKey',
            'Subset',
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
