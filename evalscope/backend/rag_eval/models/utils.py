"""Shared utilities for model loading and downloading."""
import os
from typing import Optional

from evalscope.constants import HubType
from evalscope.utils.logger import get_logger

logger = get_logger()


def download_model(model_id: str, revision: Optional[str] = 'master', hub: str = HubType.MODELSCOPE) -> str:
    """Download a model from ModelScope or HuggingFace hub.

    Args:
        model_id: The model identifier (e.g. 'BAAI/bge-large-zh-v1.5').
        revision: The model revision/branch to download.
        hub: The hub to download from ('modelscope' or 'huggingface').

    Returns:
        Local path to the downloaded model.
    """
    if hub == HubType.MODELSCOPE:
        from modelscope import snapshot_download
        logger.info(f'Downloading model {model_id} from ModelScope (revision={revision})')
        model_path = snapshot_download(model_id=model_id, revision=revision)
    else:
        from huggingface_hub import snapshot_download as hf_snapshot_download
        logger.info(f'Downloading model {model_id} from HuggingFace (revision={revision})')
        model_path = hf_snapshot_download(repo_id=model_id, revision=revision)
    return model_path


def resolve_model_path(
    model_name_or_path: str, hub: str = HubType.MODELSCOPE, revision: Optional[str] = 'master'
) -> str:
    """Resolve model name to local path, downloading if necessary.

    Args:
        model_name_or_path: Local path or remote model identifier.
        hub: The hub to download from if model needs downloading.
        revision: The model revision/branch.

    Returns:
        Local path to the model.
    """
    if os.path.exists(model_name_or_path):
        return model_name_or_path
    return download_model(model_name_or_path, revision=revision, hub=hub)
