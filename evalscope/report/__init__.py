# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .combinator import (
        gen_perf_table,
        gen_table,
        get_data_frame,
        get_display_data_frame,
        get_report_list,
        percentage_weighted_average_from_subsets,
        unweighted_average_from_subsets,
        weighted_average_from_subsets,
    )
    from .generator import ReportGenerator
    from .ref import ReportRef
    from .renderer import gen_html_report_file
    from .report import (
        BenchmarkAnalysisContext,
        Category,
        ExecutionSubset,
        ExecutionSummary,
        Metric,
        Report,
        ReportKey,
        Subset,
        build_analysis_context,
    )

else:
    _import_structure = {
        'combinator': [
            'gen_perf_table',
            'gen_table',
            'get_data_frame',
            'get_display_data_frame',
            'get_report_list',
            'weighted_average_from_subsets',
            'unweighted_average_from_subsets',
            'percentage_weighted_average_from_subsets',
        ],
        'generator': [
            'ReportGenerator',
        ],
        'ref': [
            'ReportRef',
        ],
        'report': [
            'Category',
            'BenchmarkAnalysisContext',
            'ExecutionSubset',
            'ExecutionSummary',
            'Report',
            'ReportKey',
            'Subset',
            'Metric',
            'build_analysis_context',
        ],
        'renderer': [
            'gen_html_report_file',
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
