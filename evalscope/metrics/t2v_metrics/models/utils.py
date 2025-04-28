import os
from modelscope import snapshot_download


def download_open_clip_model(model_name, tag, cache_dir):
    import open_clip

    # get pretrained config
    pretrained_cfg = open_clip.get_pretrained_cfg(model_name, tag)
    model_hub = pretrained_cfg.get('hf_hub').strip('/')
    # load model from modelscope
    model_weight_name = 'open_clip_model.safetensors'
    local_path = snapshot_download(model_id=model_hub, cache_dir=cache_dir, allow_patterns=model_weight_name)
    model_file_path = os.path.join(local_path, model_weight_name)

    return model_file_path


def download_file(model_id, file_name=None, cache_dir=None):
    # download file from modelscope
    local_path = snapshot_download(model_id=model_id, cache_dir=cache_dir, allow_patterns=file_name)
    if file_name is None:
        return local_path
    else:
        return os.path.join(local_path, file_name)
