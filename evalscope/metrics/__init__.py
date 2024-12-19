# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.metrics.metrics import exact_match, weighted_mean

WeightedAverageAccuracy = {'name': 'WeightedAverageAccuracy', 'object': weighted_mean}
