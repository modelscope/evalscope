from modelscope import snapshot_download
from evalscope.utils.logger import get_logger

logger = get_logger()


def download_model(model_id: str, revision: str):
    """
    default base dir: '~/.cache/modelscope/hub/model_id'
    """
    revision = revision or "master"

    logger.info(f"Loading model {model_id} from modelscope")

    model_path = snapshot_download(model_id=model_id, revision=revision)

    return model_path, revision
