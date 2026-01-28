# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from evalscope.api.dataset import Sample


@dataclass
class ImageStatistics:
    """Statistics for image modality in the dataset."""

    count_total: int = 0
    """Total number of images across all samples."""

    count_per_sample_min: int = 0
    """Minimum number of images per sample."""

    count_per_sample_max: int = 0
    """Maximum number of images per sample."""

    count_per_sample_mean: float = 0.0
    """Average number of images per sample."""

    resolutions: List[str] = field(default_factory=list)
    """List of unique resolutions found (e.g., '1920x1080')."""

    resolution_min: Optional[str] = None
    """Minimum resolution (by total pixels)."""

    resolution_max: Optional[str] = None
    """Maximum resolution (by total pixels)."""

    formats: List[str] = field(default_factory=list)
    """List of unique image formats found (e.g., 'jpeg', 'png')."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'count_total': self.count_total,
            'count_per_sample': {
                'min': self.count_per_sample_min,
                'max': self.count_per_sample_max,
                'mean': round(self.count_per_sample_mean, 2) if self.count_per_sample_mean else 0,
            },
            'resolutions': self.resolutions[:10] if len(self.resolutions) > 10 else self.resolutions,
            'resolution_range': {
                'min': self.resolution_min,
                'max': self.resolution_max,
            } if self.resolution_min or self.resolution_max else None,
            'formats': self.formats,
        }


@dataclass
class AudioStatistics:
    """Statistics for audio modality in the dataset."""

    count_total: int = 0
    """Total number of audio files across all samples."""

    count_per_sample_min: int = 0
    """Minimum number of audio files per sample."""

    count_per_sample_max: int = 0
    """Maximum number of audio files per sample."""

    count_per_sample_mean: float = 0.0
    """Average number of audio files per sample."""

    duration_min: Optional[float] = None
    """Minimum audio duration in seconds."""

    duration_max: Optional[float] = None
    """Maximum audio duration in seconds."""

    duration_mean: Optional[float] = None
    """Average audio duration in seconds."""

    sample_rates: List[int] = field(default_factory=list)
    """List of unique sample rates found (e.g., 44100, 48000)."""

    formats: List[str] = field(default_factory=list)
    """List of unique audio formats found (e.g., 'mp3', 'wav')."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'count_total': self.count_total,
            'count_per_sample': {
                'min': self.count_per_sample_min,
                'max': self.count_per_sample_max,
                'mean': round(self.count_per_sample_mean, 2) if self.count_per_sample_mean else 0,
            },
            'duration': {
                'min': round(self.duration_min, 2) if self.duration_min else None,
                'max': round(self.duration_max, 2) if self.duration_max else None,
                'mean': round(self.duration_mean, 2) if self.duration_mean else None,
            } if self.duration_min is not None else None,
            'sample_rates': self.sample_rates,
            'formats': self.formats,
        }


@dataclass
class VideoStatistics:
    """Statistics for video modality in the dataset."""

    count_total: int = 0
    """Total number of videos across all samples."""

    count_per_sample_min: int = 0
    """Minimum number of videos per sample."""

    count_per_sample_max: int = 0
    """Maximum number of videos per sample."""

    count_per_sample_mean: float = 0.0
    """Average number of videos per sample."""

    duration_min: Optional[float] = None
    """Minimum video duration in seconds."""

    duration_max: Optional[float] = None
    """Maximum video duration in seconds."""

    duration_mean: Optional[float] = None
    """Average video duration in seconds."""

    resolutions: List[str] = field(default_factory=list)
    """List of unique resolutions found (e.g., '1920x1080')."""

    formats: List[str] = field(default_factory=list)
    """List of unique video formats found (e.g., 'mp4', 'mov')."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'count_total': self.count_total,
            'count_per_sample': {
                'min': self.count_per_sample_min,
                'max': self.count_per_sample_max,
                'mean': round(self.count_per_sample_mean, 2) if self.count_per_sample_mean else 0,
            },
            'duration': {
                'min': round(self.duration_min, 2) if self.duration_min else None,
                'max': round(self.duration_max, 2) if self.duration_max else None,
                'mean': round(self.duration_mean, 2) if self.duration_mean else None,
            } if self.duration_min is not None else None,
            'resolutions': self.resolutions[:10] if len(self.resolutions) > 10 else self.resolutions,
            'formats': self.formats,
        }


@dataclass
class MultimodalStatistics:
    """Combined statistics for all modalities in the dataset."""

    has_images: bool = False
    """Whether the dataset contains images."""

    has_audio: bool = False
    """Whether the dataset contains audio."""

    has_video: bool = False
    """Whether the dataset contains video."""

    image_stats: Optional[ImageStatistics] = None
    """Image modality statistics."""

    audio_stats: Optional[AudioStatistics] = None
    """Audio modality statistics."""

    video_stats: Optional[VideoStatistics] = None
    """Video modality statistics."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            'has_images': self.has_images,
            'has_audio': self.has_audio,
            'has_video': self.has_video,
        }
        if self.image_stats:
            result['image'] = self.image_stats.to_dict()
        if self.audio_stats:
            result['audio'] = self.audio_stats.to_dict()
        if self.video_stats:
            result['video'] = self.video_stats.to_dict()
        return result

    def to_markdown_table(self) -> str:
        """Generate a markdown table representation of multimodal statistics."""
        lines = []

        if self.has_images and self.image_stats:
            lines.append('**Image Statistics:**')
            lines.append('')
            lines.append('| Metric | Value |')
            lines.append('|--------|-------|')
            lines.append(f'| Total Images | {self.image_stats.count_total:,} |')
            lines.append(
                f'| Images per Sample | min: {self.image_stats.count_per_sample_min}, '
                f'max: {self.image_stats.count_per_sample_max}, '
                f'mean: {self.image_stats.count_per_sample_mean:.2f} |'
            )
            if self.image_stats.resolution_min or self.image_stats.resolution_max:
                lines.append(
                    f'| Resolution Range | {self.image_stats.resolution_min or "N/A"} - '
                    f'{self.image_stats.resolution_max or "N/A"} |'
                )
            if self.image_stats.formats:
                lines.append(f'| Formats | {", ".join(self.image_stats.formats)} |')
            lines.append('')

        if self.has_audio and self.audio_stats:
            lines.append('**Audio Statistics:**')
            lines.append('')
            lines.append('| Metric | Value |')
            lines.append('|--------|-------|')
            lines.append(f'| Total Audio Files | {self.audio_stats.count_total:,} |')
            lines.append(
                f'| Audio per Sample | min: {self.audio_stats.count_per_sample_min}, '
                f'max: {self.audio_stats.count_per_sample_max}, '
                f'mean: {self.audio_stats.count_per_sample_mean:.2f} |'
            )
            if self.audio_stats.duration_mean is not None:
                lines.append(
                    f'| Duration (sec) | min: {self.audio_stats.duration_min:.2f}, '
                    f'max: {self.audio_stats.duration_max:.2f}, '
                    f'mean: {self.audio_stats.duration_mean:.2f} |'
                )
            if self.audio_stats.sample_rates:
                lines.append(f'| Sample Rates | {", ".join(map(str, self.audio_stats.sample_rates))} Hz |')
            if self.audio_stats.formats:
                lines.append(f'| Formats | {", ".join(self.audio_stats.formats)} |')
            lines.append('')

        if self.has_video and self.video_stats:
            lines.append('**Video Statistics:**')
            lines.append('')
            lines.append('| Metric | Value |')
            lines.append('|--------|-------|')
            lines.append(f'| Total Videos | {self.video_stats.count_total:,} |')
            lines.append(
                f'| Videos per Sample | min: {self.video_stats.count_per_sample_min}, '
                f'max: {self.video_stats.count_per_sample_max}, '
                f'mean: {self.video_stats.count_per_sample_mean:.2f} |'
            )
            if self.video_stats.duration_mean is not None:
                lines.append(
                    f'| Duration (sec) | min: {self.video_stats.duration_min:.2f}, '
                    f'max: {self.video_stats.duration_max:.2f}, '
                    f'mean: {self.video_stats.duration_mean:.2f} |'
                )
            if self.video_stats.formats:
                lines.append(f'| Formats | {", ".join(self.video_stats.formats)} |')
            lines.append('')

        return '\n'.join(lines)


@dataclass
class SubsetStatistics:
    """Statistics for a single subset of a benchmark dataset."""

    name: str
    """Name of the subset."""

    sample_count: int
    """Number of samples in this subset."""

    prompt_length_mean: float = 0.0
    """Mean prompt length in characters."""

    prompt_length_min: int = 0
    """Minimum prompt length in characters."""

    prompt_length_max: int = 0
    """Maximum prompt length in characters."""

    prompt_length_std: Optional[float] = None
    """Standard deviation of prompt length."""

    target_length_mean: Optional[float] = None
    """Mean target/answer length in characters (if applicable)."""

    multimodal: Optional[MultimodalStatistics] = None
    """Multimodal statistics for this subset."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            'name': self.name,
            'sample_count': self.sample_count,
            'prompt_length_mean': round(self.prompt_length_mean, 2) if self.prompt_length_mean else 0,
            'prompt_length_min': self.prompt_length_min,
            'prompt_length_max': self.prompt_length_max,
            'prompt_length_std': round(self.prompt_length_std, 2) if self.prompt_length_std else None,
            'target_length_mean': round(self.target_length_mean, 2) if self.target_length_mean else None,
        }
        if self.multimodal:
            result['multimodal'] = self.multimodal.to_dict()
        return result


@dataclass
class DataStatistics:
    """
    Statistics about a benchmark dataset.

    This class holds computed statistics about the dataset including sample counts,
    prompt lengths, and per-subset breakdowns.
    """

    total_samples: int = 0
    """Total number of samples in the evaluation split."""

    subset_stats: List[SubsetStatistics] = field(default_factory=list)
    """Per-subset statistics."""

    prompt_length_mean: float = 0.0
    """Mean prompt length across all samples (in characters)."""

    prompt_length_min: int = 0
    """Minimum prompt length across all samples."""

    prompt_length_max: int = 0
    """Maximum prompt length across all samples."""

    prompt_length_std: Optional[float] = None
    """Standard deviation of prompt length."""

    target_length_mean: Optional[float] = None
    """Mean target/answer length (if applicable)."""

    multimodal: Optional[MultimodalStatistics] = None
    """Aggregated multimodal statistics across all subsets."""

    computed_at: Optional[str] = None
    """ISO timestamp when statistics were computed."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            'total_samples': self.total_samples,
            'subset_stats': [s.to_dict() for s in self.subset_stats],
            'prompt_length': {
                'mean': round(self.prompt_length_mean, 2) if self.prompt_length_mean else 0,
                'min': self.prompt_length_min,
                'max': self.prompt_length_max,
                'std': round(self.prompt_length_std, 2) if self.prompt_length_std else None,
            },
            'target_length_mean': round(self.target_length_mean, 2) if self.target_length_mean else None,
            'computed_at': self.computed_at,
        }
        if self.multimodal:
            result['multimodal'] = self.multimodal.to_dict()
        return result

    def to_markdown_table(self, include_overall: bool = True) -> str:
        """Generate a markdown table representation of the statistics."""
        lines = []

        if include_overall:
            lines.append('| Metric | Value |')
            lines.append('|--------|-------|')
            lines.append(f'| Total Samples | {self.total_samples:,} |')
            lines.append(f'| Prompt Length (Mean) | {self.prompt_length_mean:.1f} chars |')
            lines.append(f'| Prompt Length (Min/Max) | {self.prompt_length_min} / {self.prompt_length_max} chars |')
            if self.prompt_length_std:
                lines.append(f'| Prompt Length (Std) | {self.prompt_length_std:.1f} |')
            if self.target_length_mean:
                lines.append(f'| Target Length (Mean) | {self.target_length_mean:.1f} chars |')

        if self.subset_stats and len(self.subset_stats) > 1:
            lines.append('')
            lines.append('**Per-Subset Statistics:**')
            lines.append('')
            lines.append('| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |')
            lines.append('|--------|---------|-------------|------------|------------|')
            for s in self.subset_stats:
                lines.append(
                    f'| `{s.name}` | {s.sample_count:,} | {s.prompt_length_mean:.1f} | '
                    f'{s.prompt_length_min} | {s.prompt_length_max} |'
                )

        # Add multimodal statistics if present
        if self.multimodal and (self.multimodal.has_images or self.multimodal.has_audio or self.multimodal.has_video):
            lines.append('')
            lines.append(self.multimodal.to_markdown_table())

        return '\n'.join(lines)


# Default max lengths for truncation
DEFAULT_MAX_LENGTH = 500
DEFAULT_MAX_LIST_ITEMS = 10  # Maximum number of items to show in a list before truncating

# Base64 data URI pattern
BASE64_DATA_URI_PATTERN = r'^data:([^;,]+)(?:;[^,]*)?,'


def _format_base64_placeholder(data_uri: str) -> str:
    """
    Convert a base64 data URI to a readable placeholder.

    Args:
        data_uri: A base64 data URI string.

    Returns:
        A placeholder string like '[BASE64_IMAGE: jpeg, ~50KB]'
    """
    import re

    match = re.match(BASE64_DATA_URI_PATTERN, data_uri)
    if not match:
        return '[BASE64_DATA]'

    mime_type = match.group(1)
    # Extract type and subtype (e.g., 'image/jpeg' -> 'IMAGE', 'jpeg')
    parts = mime_type.split('/')
    if len(parts) == 2:
        media_type = parts[0].upper()
        subtype = parts[1].lower()
    else:
        media_type = 'DATA'
        subtype = mime_type.lower()

    # Estimate size from base64 length (base64 is ~4/3 of original)
    base64_start = data_uri.find(',') + 1
    if base64_start > 0:
        base64_len = len(data_uri) - base64_start
        estimated_bytes = int(base64_len * 3 / 4)
        if estimated_bytes >= 1024 * 1024:
            size_str = f'~{estimated_bytes / (1024 * 1024):.1f}MB'
        elif estimated_bytes >= 1024:
            size_str = f'~{estimated_bytes / 1024:.1f}KB'
        else:
            size_str = f'~{estimated_bytes}B'
    else:
        size_str = 'unknown size'

    return f'[BASE64_{media_type}: {subtype}, {size_str}]'


def _is_base64_data_uri(value: str) -> bool:
    """Check if a string is a base64 data URI."""
    return value.startswith('data:') and ';base64,' in value[:100]


def truncate_value(
    value: Any, max_length: int = DEFAULT_MAX_LENGTH, max_list_items: int = DEFAULT_MAX_LIST_ITEMS
) -> Any:
    """
    Recursively truncate values that are too long.

    For base64 data URIs, replaces with a readable placeholder.
    For other strings, truncates from the middle, keeping beginning and end.
    For long lists, limits the number of items shown.

    Args:
        value: The value to truncate.
        max_length: Maximum length for string values.
        max_list_items: Maximum number of items to show in a list.

    Returns:
        Truncated value with same type structure.
    """
    if value is None:
        return None
    elif isinstance(value, str):
        # Special handling for base64 data URIs
        if _is_base64_data_uri(value):
            return _format_base64_placeholder(value)
        # Regular truncation for long strings
        if len(value) > max_length:
            keep_each = (max_length - 15) // 2  # 15 chars for "... [TRUNCATED] "
            return value[:keep_each] + f' ... [TRUNCATED {len(value) - keep_each * 2} chars] ... ' + value[-keep_each:]
        return value
    elif isinstance(value, dict):
        return {k: truncate_value(v, max_length, max_list_items) for k, v in value.items()}
    elif isinstance(value, list):
        # Truncate long lists to show only first few items
        if len(value) > max_list_items:
            truncated_list = [truncate_value(item, max_length, max_list_items) for item in value[:max_list_items]]
            truncated_list.append(f'... [TRUNCATED {len(value) - max_list_items} more items] ...')
            return truncated_list
        return [truncate_value(item, max_length, max_list_items) for item in value]
    elif isinstance(value, (int, float, bool)):
        return value
    else:
        # For other types, convert to string and truncate
        str_val = str(value)
        if len(str_val) > max_length:
            keep_each = (max_length - 15) // 2
            return str_val[:keep_each] + ' ... [TRUNCATED] ... ' + str_val[-keep_each:]
        return str_val


@dataclass
class SampleExample:
    """
    A representative sample example for documentation.

    This class stores a truncated example from the benchmark dataset
    for display in documentation and README files. Uses JSON format
    via model_dump() with recursive truncation for long values.
    """

    data: Dict[str, Any]
    """The sample data as a dictionary (from sample.model_dump())."""

    subset: Optional[str] = None
    """Which subset this sample is from."""

    truncated: bool = False
    """Whether any content was truncated."""

    @classmethod
    def from_sample(
        cls,
        sample: Sample,
        subset: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        max_list_items: int = DEFAULT_MAX_LIST_ITEMS,
    ) -> 'SampleExample':
        """
        Create a SampleExample from a Sample object with auto-truncation.

        Uses sample.model_dump() to convert the sample to a dictionary,
        then recursively truncates any values that exceed max_length or max_list_items.

        Args:
            sample: The Sample object to convert.
            subset: Optional subset name override.
            max_length: Maximum length for string values before truncation.
            max_list_items: Maximum number of items to show in a list.

        Returns:
            SampleExample with truncated content if necessary.
        """
        # Get raw data from sample using model_dump
        raw_data = sample.model_dump(exclude_unset=True, exclude_none=True)

        # Check if truncation is needed
        truncated_data = truncate_value(raw_data, max_length, max_list_items)
        truncated_str = str(truncated_data)
        was_truncated = '[TRUNCATED]' in truncated_str

        return cls(
            data=truncated_data,
            subset=subset or sample.subset_key,
            truncated=was_truncated,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'data': self.data,
            'subset': self.subset,
            'truncated': self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SampleExample':
        """Create from a dictionary (for deserialization)."""
        return cls(
            data=data.get('data', {}),
            subset=data.get('subset'),
            truncated=data.get('truncated', False),
        )

    def to_json_block(self, indent: int = 2) -> str:
        """
        Generate a JSON code block representation of the sample.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON code block string for markdown display.
        """
        import json

        json_str = json.dumps(self.data, ensure_ascii=False, indent=indent)
        lines = []
        if self.subset:
            lines.append(f'**Subset**: `{self.subset}`')
            lines.append('')
        lines.append('```json')
        lines.append(json_str)
        lines.append('```')
        if self.truncated:
            lines.append('')
            lines.append('*Note: Some content was truncated for display.*')
        return '\n'.join(lines)
