import base64
import io
import os
from modelscope import snapshot_download

from evalscope.utils.logger import get_logger

logger = get_logger()


def PIL_to_bytes(image_format, **kwargs):
    OPTIONS = {
        'webp': dict(format='webp', lossless=True),
        'png': dict(format='png'),
        'jpg': dict(format='jpeg'),
    }

    def transform(image):
        bytestream = io.BytesIO()
        image.save(bytestream, **OPTIONS[image_format])
        return bytestream.getvalue()

    return transform


def PIL_to_base64(image, **kwargs):
    bytestream = io.BytesIO()
    image.save(bytestream, format='jpeg')
    return base64.b64encode(bytestream.getvalue()).decode('utf-8')


def path_to_bytes(filepath):
    with open(filepath, 'rb') as fp:
        return fp.read()


def path_to_base64(filepath):
    file_content = path_to_bytes(filepath)
    return base64.b64encode(file_content).decode('utf-8')


def ensure_dir(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def save_to_jsonl(df, file_path):
    ensure_dir(file_path)
    df.to_json(file_path, orient='records', lines=True, force_ascii=False)


def save_to_tsv(df, file_path):
    ensure_dir(file_path)
    df.to_csv(file_path, sep='\t', index=False)


def download_model(model_id: str, revision: str):
    """
    default base dir: '~/.cache/modelscope/hub/model_id'
    """
    logger.info(f'Loading model {model_id} from modelscope')

    model_path = snapshot_download(model_id=model_id, revision=revision)

    return model_path
