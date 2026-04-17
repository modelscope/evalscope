import base64
import csv
import hashlib
import io
import json
import jsonlines as jsonl
import numpy as np
import os
import re
import string
import unicodedata
import yaml
from datetime import datetime
from io import BytesIO
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.constants import DumpMode
from evalscope.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    """Create *path* and all intermediate directories if they do not exist.

    Args:
        path (str): Directory path to create.  Empty strings are silently
            ignored so callers need not guard against ``os.path.dirname``
            returning ``''`` for a bare filename.
    """
    if path:
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Output directory structure
# ---------------------------------------------------------------------------


class OutputsStructure:
    """Lazy directory structure for benchmark output artefacts.

    Each directory property is created on first access so that no empty
    folders are left behind when a particular output type is not used.

    Attributes:
        LOGS_DIR: Sub-directory name for log files.
        PREDICTIONS_DIR: Sub-directory name for prediction files.
        REVIEWS_DIR: Sub-directory name for review files.
        REPORTS_DIR: Sub-directory name for report files.
        CONFIGS_DIR: Sub-directory name for configuration snapshots.
    """

    LOGS_DIR = 'logs'
    PREDICTIONS_DIR = 'predictions'
    REVIEWS_DIR = 'reviews'
    REPORTS_DIR = 'reports'
    CONFIGS_DIR = 'configs'

    def __init__(self, outputs_dir: str, is_make: bool = True) -> None:
        """Initialize the output directory structure.

        Args:
            outputs_dir (str): Root output directory.
            is_make (bool): Whether to create sub-directories on first
                access.  Defaults to ``True``.
        """
        self.outputs_dir = outputs_dir
        self.is_make = is_make
        self._dirs: Dict[str, Optional[str]] = {
            'logs_dir': None,
            'predictions_dir': None,
            'reviews_dir': None,
            'reports_dir': None,
            'configs_dir': None,
        }

    def _get_dir(self, attr_name: str, dir_name: str) -> str:
        """Return (and optionally create) a sub-directory path.

        Args:
            attr_name (str): Cache key within ``self._dirs``.
            dir_name (str): Sub-directory name relative to
                ``self.outputs_dir``.

        Returns:
            str: Absolute path to the sub-directory.
        """
        if self._dirs[attr_name] is None:
            dir_path = os.path.join(self.outputs_dir, dir_name)
            if self.is_make:
                _ensure_dir(dir_path)
            self._dirs[attr_name] = dir_path
        return self._dirs[attr_name]  # type: ignore[return-value]

    @property
    def logs_dir(self) -> str:
        """Path to the logs sub-directory."""
        return self._get_dir('logs_dir', OutputsStructure.LOGS_DIR)

    @property
    def predictions_dir(self) -> str:
        """Path to the predictions sub-directory."""
        return self._get_dir('predictions_dir', OutputsStructure.PREDICTIONS_DIR)

    @property
    def reviews_dir(self) -> str:
        """Path to the reviews sub-directory."""
        return self._get_dir('reviews_dir', OutputsStructure.REVIEWS_DIR)

    @property
    def reports_dir(self) -> str:
        """Path to the reports sub-directory."""
        return self._get_dir('reports_dir', OutputsStructure.REPORTS_DIR)

    @property
    def configs_dir(self) -> str:
        """Path to the configs sub-directory."""
        return self._get_dir('configs_dir', OutputsStructure.CONFIGS_DIR)


# ---------------------------------------------------------------------------
# JSONL utilities
# ---------------------------------------------------------------------------


def jsonl_to_list(jsonl_file: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Attempts to use the ``jsonlines`` library first; falls back to
    line-by-line ``json.loads`` parsing on any error.

    Args:
        jsonl_file (str): Path to the ``.jsonl`` file.

    Returns:
        List[Dict[str, Any]]: Parsed records.  Returns an empty list and
        logs a warning when the file contains no valid records.
    """
    res_list: List[Dict[str, Any]] = []
    try:
        with jsonl.open(jsonl_file, mode='r') as reader:
            for line in reader.iter(type=dict, allow_none=True, skip_invalid=False):
                res_list.append(line)
    except Exception:
        # Fallback: parse line-by-line with the stdlib json module.
        res_list = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    res_list.append(json.loads(stripped))

    if not res_list:
        logger.warning(f'No data found in {jsonl_file}.')
    return res_list


def jsonl_to_reader(jsonl_file: str) -> jsonl.Reader:
    """Open a JSONL file and return a :class:`jsonlines.Reader` object.

    .. warning::
        The caller is responsible for closing the returned reader.  Use it
        as a context manager (``with jsonl_to_reader(...) as r:``) to
        ensure the underlying file is released when done.

    Args:
        jsonl_file (str): Path to the ``.jsonl`` file.

    Returns:
        jsonlines.Reader: An open reader for the file.
    """
    return jsonl.open(jsonl_file, mode='r')


def dump_jsonl_data(
    data_list: Union[List[Any], Any],
    jsonl_file: str,
    dump_mode: DumpMode = DumpMode.OVERWRITE,
) -> None:
    """Serialize *data_list* to a JSONL file.

    Args:
        data_list (Union[List[Any], Any]): Data to write.  A single
            non-list value is automatically wrapped in a list.
        jsonl_file (str): Destination file path.  Parent directories are
            created automatically.
        dump_mode (DumpMode): :attr:`~DumpMode.OVERWRITE` (default) to
            replace an existing file, or :attr:`~DumpMode.APPEND` to add
            records to the end.

    Raises:
        ValueError: When *jsonl_file* is empty or ``None``.
    """
    if not jsonl_file:
        raise ValueError('output file must be provided.')

    jsonl_file = os.path.expanduser(jsonl_file)
    _ensure_dir(os.path.dirname(jsonl_file))

    if not isinstance(data_list, list):
        data_list = [data_list]

    data_list = convert_normal_types(data_list)

    mode = 'w' if dump_mode == DumpMode.OVERWRITE else 'a'
    with jsonl.open(jsonl_file, mode=mode) as writer:
        writer.write_all(data_list)


# ---------------------------------------------------------------------------
# CSV / TSV utilities
# ---------------------------------------------------------------------------


def jsonl_to_csv(jsonl_file: str, csv_file: str) -> None:
    """Convert a JSONL file to CSV format.

    Args:
        jsonl_file (str): Source ``.jsonl`` file path.
        csv_file (str): Destination ``.csv`` file path.  Parent directories
            are created automatically.
    """
    data = jsonl_to_list(jsonl_file)
    if not data:
        logger.warning(f'No data found in {jsonl_file}.')
        return

    _ensure_dir(os.path.dirname(csv_file))
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data[0].keys())
        for item in data:
            writer.writerow(item.values())


def csv_to_list(csv_file: str) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dicts.

    Args:
        csv_file (str): Path to the ``.csv`` file.

    Returns:
        List[Dict[str, str]]: One dict per row, keyed by column headers.

    Raises:
        ValueError: When the file cannot be read or parsed.
    """
    try:
        res_list: List[Dict[str, str]] = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                res_list.append(dict(row))
        return res_list
    except (OSError, csv.Error) as e:
        raise ValueError(f'Failed to read CSV file "{csv_file}": {e}') from e


def tsv_to_list(tsv_file: str) -> List[Dict[str, str]]:
    """Read a TSV (tab-separated values) file into a list of dicts.

    Args:
        tsv_file (str): Path to the ``.tsv`` file.

    Returns:
        List[Dict[str, str]]: One dict per row, keyed by column headers.

    Raises:
        ValueError: When the file cannot be read or parsed.
    """
    try:
        res_list: List[Dict[str, str]] = []
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                res_list.append(dict(row))
        return res_list
    except (OSError, csv.Error) as e:
        raise ValueError(f'Failed to read TSV file "{tsv_file}": {e}') from e


def csv_to_jsonl(csv_file: str, jsonl_file: str) -> None:
    """Convert a CSV file to JSONL format.

    Args:
        csv_file (str): Source ``.csv`` file path.
        jsonl_file (str): Destination ``.jsonl`` file path.
    """
    data = csv_to_list(csv_file)
    if not data:
        logger.warning(f'No data found in {csv_file}.')
        return
    dump_jsonl_data(data, jsonl_file, dump_mode=DumpMode.OVERWRITE)


# ---------------------------------------------------------------------------
# YAML / JSON utilities
# ---------------------------------------------------------------------------


def yaml_to_dict(yaml_file: str) -> Any:
    """Load a YAML file and return its contents as a Python object.

    Args:
        yaml_file (str): Path to the ``.yaml`` / ``.yml`` file.

    Returns:
        Any: Parsed YAML content (typically a ``dict`` or ``list``).

    Raises:
        yaml.YAMLError: When the file cannot be parsed.
    """
    with open(yaml_file, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f'Failed to parse YAML file "{yaml_file}": {e}')
            raise


def dict_to_yaml(d: dict, yaml_file: str) -> None:
    """Dump a dict to a YAML file.

    Args:
        d (dict): Data to serialize.
        yaml_file (str): Destination file path.  Parent directories are
            created automatically.
    """
    _ensure_dir(os.path.dirname(yaml_file))
    with open(yaml_file, 'w') as f:
        yaml.dump(d, f, default_flow_style=False, allow_unicode=True, Dumper=yaml.SafeDumper)


def json_to_dict(json_file: str) -> Any:
    """Load a JSON file and return its contents as a Python object.

    Args:
        json_file (str): Path to the ``.json`` file.

    Returns:
        Any: Parsed JSON content.

    Raises:
        json.JSONDecodeError: When the file cannot be parsed.
    """
    with open(json_file, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse JSON file "{json_file}": {e}')
            raise


def dict_to_json(d: dict, json_file: str) -> None:
    """Dump a dict to a JSON file with pretty-printing.

    Args:
        d (dict): Data to serialize.
        json_file (str): Destination file path.  Parent directories are
            created automatically.
    """
    json_file = os.path.expanduser(json_file)
    _ensure_dir(os.path.dirname(json_file))
    with open(json_file, 'w') as f:
        json.dump(d, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Path / filesystem utilities
# ---------------------------------------------------------------------------


def are_paths_same(path1: str, path2: str) -> bool:
    """Return ``True`` when *path1* and *path2* resolve to the same inode.

    Symlinks, relative components, and ``~`` expansions are all resolved
    before comparison.

    Args:
        path1 (str): First path.
        path2 (str): Second path.

    Returns:
        bool: ``True`` if both paths point to the same file-system entry.
    """
    real_path1 = os.path.realpath(os.path.abspath(os.path.expanduser(path1)))
    real_path2 = os.path.realpath(os.path.abspath(os.path.expanduser(path2)))
    return real_path1 == real_path2


def get_latest_folder_path(work_dir: str) -> Optional[str]:
    """Return the path to the most recently created timestamped sub-folder.

    Only sub-folders whose names match the ``YYYYMMDD_HHMMSS`` pattern are
    considered.

    Args:
        work_dir (str): Parent directory to search.

    Returns:
        Optional[str]: Absolute path to the latest timestamped folder, or
        ``None`` when no matching folders are found.
    """
    timestamp_pattern = re.compile(r'^\d{8}_\d{6}$')
    folders = [
        f for f in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, f)) and timestamp_pattern.match(f)
    ]

    if not folders:
        print(f'>> No timestamped folders found in {work_dir}!')
        return None

    latest = max(folders, key=lambda name: datetime.strptime(name, '%Y%m%d_%H%M%S'))
    return os.path.join(work_dir, latest)


def safe_filename(s: str, max_length: int = 255) -> str:
    """Convert an arbitrary string into a safe file-system name.

    The function performs the following transformations in order:

    1. NFKD Unicode normalisation followed by ASCII-only encoding.
    2. Replacement of every non-alphanumeric character (except ``.``, ``-``,
       ``_``) with ``_``.
    3. Collapsing of consecutive underscores.
    4. Stripping of leading/trailing ``.`` and ``_``.
    5. Substitution of an empty result with ``'untitled'``.
    6. Prefixing with ``_`` for names that start with ``.`` (hidden files).
    7. Truncation to *max_length*, preserving the file extension if present.

    Args:
        s (str): The input string to convert.
        max_length (int): Maximum byte-length of the resulting name.
            Defaults to ``255``.

    Returns:
        str: A safe filename string.

    Examples:
        >>> safe_filename("Hello/World?.txt")
        'Hello_World.txt'
    """
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')

    safe_chars = string.ascii_letters + string.digits + '.-_'
    s = ''.join(c if c in safe_chars else '_' for c in s)
    s = re.sub(r'_+', '_', s).strip('._')

    if not s:
        s = 'untitled'

    if s.startswith('.'):
        s = '_' + s

    if len(s) > max_length:
        name, ext = os.path.splitext(s)
        ext_len = len(ext)
        if ext_len > 0:
            s = name[:max_length - ext_len] + ext
        else:
            s = s[:max_length]

    return s


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def gen_hash(name: str, bits: int = 32) -> str:
    """Return the first *bits* hex characters of the MD5 hash of *name*.

    Args:
        name (str): Input string to hash.
        bits (int): Number of hex characters to return.  Defaults to ``32``
            (the full MD5 digest).

    Returns:
        str: Hex digest substring of length *bits*.
    """
    return hashlib.md5(name.encode(encoding='UTF-8')).hexdigest()[:bits]


# ---------------------------------------------------------------------------
# List utilities
# ---------------------------------------------------------------------------


def get_valid_list(
    input_list: List[Any],
    candidate_list: List[Any],
) -> Tuple[List[Any], List[Any]]:
    """Partition *input_list* into elements that are (or are not) in *candidate_list*.

    Args:
        input_list (List[Any]): The list to partition.
        candidate_list (List[Any]): The reference set of valid values.

    Returns:
        Tuple[List[Any], List[Any]]: A 2-tuple of
        ``(valid_list, invalid_list)`` where *valid_list* contains every
        element of *input_list* found in *candidate_list* and *invalid_list*
        contains the rest.
    """
    candidate_set = set(candidate_list)
    valid = [i for i in input_list if i in candidate_set]
    invalid = [i for i in input_list if i not in candidate_set]
    return valid, invalid


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def PIL_to_base64(image: Image.Image, format: str = 'JPEG', add_header: bool = False) -> str:
    """Encode a PIL Image as a base64 string.

    Args:
        image (Image.Image): The source PIL Image.
        format (str): Target image format (e.g. ``'JPEG'``, ``'PNG'``).
            Defaults to ``'JPEG'``.
        add_header (bool): When ``True``, prepend the data-URI header
            ``data:image/<format>;base64,``.  Defaults to ``False``.

    Returns:
        str: Base64-encoded image string, optionally with a data-URI header.
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    if add_header:
        img_str = f'data:image/{format.lower()};base64,{img_str}'
    return img_str


def bytes_to_base64(
    bytes_data: bytes,
    *,
    format: str = 'png',
    add_header: bool = False,
    content_type: str = 'image',
) -> str:
    """Encode raw bytes as a base64 string.

    Args:
        bytes_data (bytes): The raw bytes to encode.
        format (str): MIME sub-type of the data (e.g. ``'png'``, ``'wav'``).
            Used only when *add_header* is ``True``.  Defaults to ``'png'``.
        add_header (bool): When ``True``, prepend a data-URI header of the
            form ``data:<content_type>/<format>;base64,``.  Defaults to
            ``False``.
        content_type (str): Top-level MIME type (e.g. ``'image'``,
            ``'audio'``).  Used only when *add_header* is ``True``.
            Defaults to ``'image'``.

    Returns:
        str: Base64-encoded string, optionally with a data-URI header.
    """
    base64_str = base64.b64encode(bytes_data).decode('utf-8')
    if add_header:
        base64_str = f'data:{content_type}/{format};base64,{base64_str}'
    return base64_str


def base64_to_PIL(base64_str: str) -> Image.Image:
    """Decode a base64-encoded string into a PIL Image.

    Data-URI headers (e.g. ``data:image/png;base64,``) are stripped
    automatically before decoding.

    Args:
        base64_str (str): Base64-encoded image data, with or without a
            data-URI header.

    Returns:
        Image.Image: The decoded PIL Image.
    """
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))


def convert_image_base64_format(image_base64: str) -> Tuple[str, str]:
    """Normalise an image base64 string to JPEG or PNG.

    AVIF and other exotic formats are converted to JPEG.  JPEG and PNG
    images are returned unchanged.

    Args:
        image_base64 (str): Raw base64-encoded image data (no data-URI
            header).

    Returns:
        Tuple[str, str]: A 2-tuple of ``(base64_string, format_string)``
        where *format_string* is one of ``'JPEG'`` or ``'PNG'``.  When
        conversion fails the original *image_base64* is returned together
        with the detected (or default ``'JPEG'``) format string.
    """
    image_data = base64.b64decode(image_base64)

    # Detect format via magic bytes.
    if image_data.startswith(b'\x89PNG'):
        detected_format = 'PNG'
    elif image_data.startswith(b'\xff\xd8\xff'):
        detected_format = 'JPEG'
    elif image_data[4:8] in (b'ftyp', b'avif'):
        detected_format = 'AVIF'
    else:
        detected_format = 'JPEG'
        try:
            pil_img_test = Image.open(io.BytesIO(image_data))
            if pil_img_test.format:
                detected_format = pil_img_test.format
        except Exception:
            pass

    if detected_format in ('JPEG', 'PNG'):
        return image_base64, detected_format

    # Convert non-standard formats to JPEG.
    try:
        pil_img = Image.open(io.BytesIO(image_data))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        output_buffer = io.BytesIO()
        pil_img.save(output_buffer, format='JPEG', quality=95)
        output_buffer.seek(0)
        return base64.b64encode(output_buffer.read()).decode('utf-8'), 'JPEG'
    except Exception as e:
        logger.warning(f'Failed to process or convert image (detected as {detected_format}): {e}')
        return image_base64, detected_format


def compress_image_to_limit(
    image_bytes: bytes,
    max_bytes: int = 10_000_000,
) -> Tuple[bytes, str]:
    """Ensure *image_bytes* are under *max_bytes* via JPEG re-encoding.

    The function first reduces JPEG quality in steps of 10 from 85 down to
    40, then progressively downscales the image by 10 % per iteration until
    the size constraint is satisfied or the shorter side reaches 256 pixels.

    If the original bytes are already below *max_bytes* they are returned
    unchanged with format ``'png'``.

    Args:
        image_bytes (bytes): Raw image bytes.
        max_bytes (int): Maximum allowed byte size.  Defaults to
            ``10_000_000`` (10 MB).

    Returns:
        Tuple[bytes, str]: A 2-tuple of ``(processed_bytes, format_str)``
        where *format_str* is ``'png'`` when no compression was applied and
        ``'jpeg'`` otherwise.
    """
    if len(image_bytes) <= max_bytes:
        return image_bytes, 'png'

    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as exc:
        logger.warning(f'Failed to open image bytes with PIL, sending original image; '
                       f'may exceed API limit: {exc}')
        return image_bytes, 'png'

    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    def _encode_jpeg(source: Image.Image, quality: int) -> bytes:
        buf = BytesIO()
        source.save(buf, format='JPEG', quality=quality, optimize=True, progressive=True)
        return buf.getvalue()

    quality: int = 85
    quality_floor: int = 40
    out: bytes = _encode_jpeg(img, quality)

    while len(out) > max_bytes and quality > quality_floor:
        quality -= 10
        out = _encode_jpeg(img, quality)

    min_side_floor: int = 256
    scale: float = 0.9
    while len(out) > max_bytes and min(img.size) > min_side_floor:
        new_w = max(min_side_floor, int(img.width * scale))
        new_h = max(min_side_floor, int(img.height * scale))
        if (new_w, new_h) == img.size:
            break
        img = img.resize((new_w, new_h), Image.LANCZOS)
        out = _encode_jpeg(img, quality)

    if len(out) > max_bytes:
        logger.warning(f'Image remains above limit after compression: '
                       f'size={len(out)} bytes (limit={max_bytes}).')
    else:
        logger.info(
            f'Compressed image from {len(image_bytes)} to {len(out)} bytes; '
            f'quality={quality}, size={img.width}x{img.height}.'
        )

    return out, 'jpeg'


# ---------------------------------------------------------------------------
# Type-conversion utilities
# ---------------------------------------------------------------------------


def convert_normal_types(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable types to native Python types.

    Handles :mod:`numpy` scalars and arrays, :class:`~datetime.datetime`
    objects, and :class:`os.PathLike` instances.  Containers (``dict``,
    ``list``, ``tuple``) are traversed recursively.

    Args:
        obj (Any): The object to convert.

    Returns:
        Any: A JSON-serialisable equivalent of *obj*.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_normal_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_normal_types(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_normal_types(item) for item in obj)
    if isinstance(obj, os.PathLike):
        return str(obj)
    return obj
