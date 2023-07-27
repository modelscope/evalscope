# Copyright (c) Alibaba, Inc. and its affiliates.

from llmuses.evaluator.base_reviewer import BaseReviewer

#
# from typing import TYPE_CHECKING
#
# from llmuses.utils.import_utils import LazyImportModule
#
# if TYPE_CHECKING:
#     from .builder import build_scoring_model, SCORING_MODEL_REGISTRY
#     from .classify import ClassifyEval
#     from .generation import GenerationEval
#     from .include import IncludeEval
#     from .match import MatchEval
#     from .similarity import SimilarityEval
#     from .unit_test import UnitTestEval
# else:
#     _import_structure = {
#         'builder': ['build_scoring_model', 'SCORING_MODEL_REGISTRY'],
#         'classify': ['ClassifyEval'],
#         'generation': ['GenerationEval'],
#         'include': ['IncludeEval'],
#         'match': ['MatchEval'],
#         'similarity': ['SimilarityEval'],
#         'unit_test': ['UnitTestEval'],
#
#     }
#
#     import sys
#     sys.modules[__name__] = LazyImportModule(
#         name=__name__,
#         module_file=globals()['__file__'],
#         import_structure=_import_structure,
#     )
