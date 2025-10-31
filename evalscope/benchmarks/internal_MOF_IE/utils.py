import json
from typing import Dict, Optional
from evalscope.api.registry import get_metric
from evalscope.utils.logger import get_logger

logger = get_logger()

DEFAULT_METRIC_VALUE = {
     "anls": 0,
     "exact_match": 0
}

def calculate_metrics(data: Optional[Dict] = None, column_metric_dict: Dict ={}):
    """
    Calculate the metrics for the given data.
    1. data=None时返回默认值
    2. 根据dict选择不同的指标计算方式
    想直接用 `Mean(Aggregator)` 的话（这个是默认的Aggregator），需要在这里就把ave算好，否则比较麻烦。
    """
    metric = {"ave": 0.0}
    for k, v in column_metric_dict.items():
        if v in DEFAULT_METRIC_VALUE:
            metric[k] = DEFAULT_METRIC_VALUE[v]
        else:
            metric[k] = 0

    if not data:
        return metric
    reference = data['target']  # dict
    prediction = data['predictions']  # dict
    
    for k, v in reference.items():
        if k not in prediction:
            continue
        else:
            if v == "NaN":
                metric_scorer = get_metric("exact_match")
            elif prediction[k] == "NaN":
                continue
            else:
                metric_scorer = get_metric(column_metric_dict[k])  # Get metric implementation from registry
            metric_func = metric_scorer()  # Instantiate the metric scorer
            metric[k] = metric_func(
                prediction=prediction[k],
                reference=v,
            )
            # logger.info(f"metric is {column_metric_dict[k]}, prediction is {prediction[k]}, reference is {v}, score is {metric[k]}")

    metric["ave"] = sum(metric.values()) / len(metric.values())

    return metric
