import base64
import httpx
import mimetypes
import re


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
        bytes, mime_type = file_as_data(file)
        base64_file = base64.b64encode(bytes).decode('utf-8')
        file = f'data:{mime_type};base64,{base64_file}'
        return file
