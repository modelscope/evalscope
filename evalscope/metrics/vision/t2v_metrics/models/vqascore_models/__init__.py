from ...constants import CACHE_DIR
from .clip_t5_model import CLIP_T5_MODELS, CLIPT5Model
from .gpt4v_model import GPT4V_MODELS, GPT4VModel

ALL_VQA_MODELS = [
    CLIP_T5_MODELS,
    GPT4V_MODELS,
]


def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]


def get_vqascore_model(model_name, device='cuda', cache_dir=CACHE_DIR, **kwargs):
    assert model_name in list_all_vqascore_models()
    if model_name in CLIP_T5_MODELS:
        return CLIPT5Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in GPT4V_MODELS:
        return GPT4VModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()
