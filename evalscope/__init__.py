# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope import agent, benchmarks  # noqa: F401  # registered agent strategies; benchmark discovery
from evalscope.config import TaskConfig

# Importing the names (rather than `import *`) is what keeps `evalscope.evaluator`
# pointing at the subpackage: a star import would rebind it to the inner
# `evalscope.evaluator.evaluator` module and hide the sibling submodules.
from evalscope.evaluator import (  # registered evaluators
    BatchReviewer,
    DefaultEvaluator,
    ExecutionTracker,
    PerfCollector,
)
from evalscope.filters import extraction, selection  # registered filters
from evalscope.metrics import aggregators, audio, nlp, vision  # noqa: F401  # registered metrics & aggregators
from evalscope.models import model_apis  # need for register model apis
from evalscope.run import run_task

from .version import __release_datetime__, __version__
