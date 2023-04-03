# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Union

from evals import Eval


class GenerationEval(Eval):
    """
    GenerationEval is a subclass of Eval, which is used to evaluate the generation task of LLMs.
    """

    def __init__(self, metrics: list, predicted_samples: Union[list, str], **kwargs):
        super(GenerationEval, self).__init__(metrics, predicted_samples, **kwargs)

    def eval_samples(self):
        pass

    def run(self):
        pass
