# Copyright (c) Alibaba, Inc. and its affiliates.
from enum import Enum


class EvalBackend(Enum):
    # Use native evaluation pipeline of EvalScope
    NATIVE = 'Native'

    # Use OpenCompass framework as the evaluation backend
    OPEN_COMPASS = 'OpenCompass'
    
    # Use VLM Eval Kit as the multi-modal model evaluation backend
    VLM_EVAL_KIT = 'VLMEvalKit'
    
    # Use RAGEval as the RAG evaluation backend
    RAG_EVAL = 'RAGEval'

    # Use third-party evaluation backend/modules
    THIRD_PARTY = 'ThirdParty'
    
    

