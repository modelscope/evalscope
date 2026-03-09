import base64
import httpx
import mimetypes
import os
import re

from evalscope.utils.logger import get_logger

logger = get_logger()


def is_http_url(url: str) -> bool:
    return url.startswith('http://') or url.startswith('https://')


def is_data_uri(url: str) -> bool:
    pattern = r'^data:([^;]+);base64,.*'
    return re.match(pattern, url) is not None


def data_uri_mime_type(data_url: str) -> str | None:
    pattern = r'^data:([^;]+);.*'
    match = re.match(pattern, data_url)
    if match:
        mime_type = match.group(1)
        return mime_type
    else:
        return None


def data_uri_to_base64(data_uri: str) -> str:
    pattern = r'^data:[^,]+,'
    stripped_uri = re.sub(pattern, '', data_uri)
    return stripped_uri


def file_as_data(file: str) -> tuple[bytes, str]:
    if is_data_uri(file):
        # resolve mime type and base64 content
        mime_type = data_uri_mime_type(file) or 'image/png'
        file_base64 = data_uri_to_base64(file)
        file_bytes = base64.b64decode(file_base64)
    else:
        # guess mime type; need strict=False for webp images
        type, _ = mimetypes.guess_type(file, strict=False)
        if type:
            mime_type = type
        else:
            mime_type = 'image/png'

        # handle url or file
        if is_http_url(file):
            client = httpx.Client()
            file_bytes = client.get(file).content
        else:
            with open(file, 'rb') as f:
                file_bytes = f.read()

    # return bytes and type
    return file_bytes, mime_type


def file_as_data_uri(file: str) -> str:
    if is_data_uri(file):
        return file
    else:
        file_bytes, mime_type = file_as_data(file)
        base64_file = base64.b64encode(file_bytes).decode('utf-8')
        file = f'data:{mime_type};base64,{base64_file}'
        return file


def download_url(url: str, save_path: str, num_retries: int = 3):
    """
    Download a file from a URL to a local path with retries.

    Args:
        url (str): The URL to download from.
        save_path (str): The local file path to save the downloaded file.
        num_retries (int): Number of times to retry on failure.
    """
    import requests
    from time import sleep
    from tqdm import tqdm

    save_path = os.path.abspath(save_path)

    # Check if the file already exists before opening any network connection.
    # A lightweight HEAD request is used to fetch content-length for size verification.
    if os.path.exists(save_path):
        try:
            head = requests.head(url, timeout=10, allow_redirects=True)
            remote_size = int(head.headers.get('content-length', 0))
            if remote_size > 0 and os.path.getsize(save_path) == remote_size:
                logger.info(f'File {save_path} already exists and is complete. Skipping download.')
                return
        except Exception as e:
            logger.warning(f'HEAD request failed for {url}, will attempt full download: {e}')

    for attempt in range(num_retries):
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))

                logger.info(f'Downloading {url} to {save_path} (attempt {attempt + 1}/{num_retries})...')

                with open(save_path, 'wb') as f, tqdm(
                    desc=os.path.basename(save_path),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            size = f.write(chunk)
                            bar.update(size)
            logger.info(f'Downloaded {url} to {save_path}')
            return
        except Exception as e:
            logger.warning(f'Attempt {attempt + 1} failed to download {url}: {e}')
            if attempt < num_retries - 1:
                sleep(2**attempt)  # Exponential backoff

    raise RuntimeError(f'Failed to download {url} after {num_retries} attempts.')
