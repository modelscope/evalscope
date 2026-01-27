# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Utility functions for computing benchmark statistics and extracting sample examples.

This module provides tools to analyze benchmark datasets, compute statistics
about prompt lengths, sample counts, and extract representative examples
for documentation purposes. Supports multimodal datasets with image, audio,
and video content analysis.

Dependencies:
    - pillow: For image metadata extraction (resolution, format)
    - mutagen (optional): For audio metadata extraction (duration, sample rate, format)
"""

import base64
import io
import os
import re
import statistics as stats_module
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from evalscope.api.benchmark.statistics import (
    AudioStatistics,
    DataStatistics,
    ImageStatistics,
    MultimodalStatistics,
    SampleExample,
    SubsetStatistics,
    VideoStatistics,
)
from evalscope.api.messages import ChatMessage, ContentAudio, ContentImage, ContentVideo
from evalscope.api.messages.chat_message import ChatMessageBase
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.dataset import DatasetDict, Sample

logger = get_logger()

# Optional dependencies
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.debug('PIL not available, image metadata extraction will be limited')

try:
    import mutagen
    from mutagen.flac import FLAC
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
    from mutagen.oggvorbis import OggVorbis
    from mutagen.wave import WAVE
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False
    logger.debug('mutagen not available, audio metadata extraction will be limited')


@dataclass
class ImageInfo:
    """Information extracted from a single image."""
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    source_type: str = 'unknown'  # 'file', 'url', 'base64'


@dataclass
class AudioInfo:
    """Information extracted from a single audio file."""
    duration: Optional[float] = None  # seconds
    sample_rate: Optional[int] = None
    format: Optional[str] = None
    source_type: str = 'unknown'  # 'file', 'url', 'base64'


@dataclass
class VideoInfo:
    """Information extracted from a single video file."""
    duration: Optional[float] = None  # seconds
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    source_type: str = 'unknown'  # 'file', 'url', 'base64'


def _is_http_url(url: str) -> bool:
    """Check if a string is an HTTP/HTTPS URL."""
    return url.startswith('http://') or url.startswith('https://')


def _is_base64_data_uri(data: str) -> bool:
    """Check if a string is a base64 data URI."""
    return data.startswith('data:')


def _extract_format_from_data_uri(data_uri: str) -> Optional[str]:
    """Extract format from a data URI (e.g., 'data:image/jpeg;base64,...' -> 'jpeg')."""
    match = re.match(r'data:(?:image|audio|video)/([^;,]+)', data_uri)
    if match:
        return match.group(1).lower()
    return None


def _decode_base64_data(data_uri: str) -> Optional[bytes]:
    """Decode base64 data from a data URI."""
    try:
        base64_match = re.match(r'data:[^;]+;base64,(.+)', data_uri)
        if base64_match:
            return base64.b64decode(base64_match.group(1))
    except Exception:
        pass
    return None


def _get_image_info_from_bytes(data: bytes) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Extract image dimensions and format from bytes using PIL.

    Args:
        data: Raw image bytes.

    Returns:
        Tuple of (width, height, format).
    """
    if not HAS_PIL:
        return None, None, None

    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.width, img.height, img.format.lower() if img.format else None
    except Exception:
        return None, None, None


def _get_audio_info_from_bytes(data: bytes,
                               format_hint: Optional[str] = None
                               ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """
    Extract audio duration, sample rate, and format from bytes using mutagen.

    Args:
        data: Raw audio bytes.
        format_hint: Hint about the audio format.

    Returns:
        Tuple of (duration_seconds, sample_rate, format).
    """
    if not HAS_MUTAGEN:
        return None, None, format_hint

    try:
        audio_file = io.BytesIO(data)
        audio = mutagen.File(audio_file)
        if audio is None:
            return None, None, format_hint

        duration = audio.info.length if hasattr(audio.info, 'length') else None
        sample_rate = audio.info.sample_rate if hasattr(audio.info, 'sample_rate') else None

        # Determine format from mutagen type
        audio_format = format_hint
        if isinstance(audio, MP3):
            audio_format = 'mp3'
        elif isinstance(audio, WAVE):
            audio_format = 'wav'
        elif isinstance(audio, FLAC):
            audio_format = 'flac'
        elif isinstance(audio, OggVorbis):
            audio_format = 'ogg'
        elif isinstance(audio, MP4):
            audio_format = 'm4a'

        return duration, sample_rate, audio_format
    except Exception:
        return None, None, format_hint


def _get_audio_info_from_file(file_path: str) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """
    Extract audio info from a file path using mutagen.

    Args:
        file_path: Path to the audio file.

    Returns:
        Tuple of (duration_seconds, sample_rate, format).
    """
    if not HAS_MUTAGEN or not os.path.exists(file_path):
        _, ext = os.path.splitext(file_path)
        return None, None, ext[1:].lower() if ext else None

    try:
        audio = mutagen.File(file_path)
        if audio is None:
            _, ext = os.path.splitext(file_path)
            return None, None, ext[1:].lower() if ext else None

        duration = audio.info.length if hasattr(audio.info, 'length') else None
        sample_rate = audio.info.sample_rate if hasattr(audio.info, 'sample_rate') else None

        # Determine format
        audio_format = None
        if isinstance(audio, MP3):
            audio_format = 'mp3'
        elif isinstance(audio, WAVE):
            audio_format = 'wav'
        elif isinstance(audio, FLAC):
            audio_format = 'flac'
        elif isinstance(audio, OggVorbis):
            audio_format = 'ogg'
        elif isinstance(audio, MP4):
            audio_format = 'm4a'
        else:
            _, ext = os.path.splitext(file_path)
            audio_format = ext[1:].lower() if ext else None

        return duration, sample_rate, audio_format
    except Exception:
        _, ext = os.path.splitext(file_path)
        return None, None, ext[1:].lower() if ext else None


def get_image_info(image_source: str) -> ImageInfo:
    """
    Extract information from an image source using PIL.

    Args:
        image_source: Image URL, file path, or base64 data URI.

    Returns:
        ImageInfo with extracted metadata.
    """
    info = ImageInfo()

    if _is_http_url(image_source):
        info.source_type = 'url'
        ext_match = re.search(r'\.([a-zA-Z]+)(?:\?|$)', image_source)
        if ext_match:
            info.format = ext_match.group(1).lower()
        return info

    elif _is_base64_data_uri(image_source):
        info.source_type = 'base64'
        info.format = _extract_format_from_data_uri(image_source)

        image_data = _decode_base64_data(image_source)
        if image_data:
            width, height, fmt = _get_image_info_from_bytes(image_data)
            if width and height:
                info.width = width
                info.height = height
            if fmt:
                info.format = fmt
        return info

    else:
        # File path
        info.source_type = 'file'
        _, ext = os.path.splitext(image_source)
        if ext:
            info.format = ext[1:].lower()

        if HAS_PIL and os.path.exists(image_source):
            try:
                with Image.open(image_source) as img:
                    info.width = img.width
                    info.height = img.height
                    if img.format:
                        info.format = img.format.lower()
            except Exception:
                pass

        return info


def get_audio_info(audio_source: str, audio_format: Optional[str] = None) -> AudioInfo:
    """
    Extract information from an audio source using mutagen.

    Args:
        audio_source: Audio URL, file path, or base64 data URI.
        audio_format: Known audio format (e.g., 'mp3', 'wav').

    Returns:
        AudioInfo with extracted metadata.
    """
    info = AudioInfo()
    if audio_format:
        info.format = audio_format.lower()

    if _is_http_url(audio_source):
        info.source_type = 'url'
        if not info.format:
            ext_match = re.search(r'\.([a-zA-Z0-9]+)(?:\?|$)', audio_source)
            if ext_match:
                info.format = ext_match.group(1).lower()
        return info

    elif _is_base64_data_uri(audio_source):
        info.source_type = 'base64'
        if not info.format:
            info.format = _extract_format_from_data_uri(audio_source)

        audio_data = _decode_base64_data(audio_source)
        if audio_data:
            duration, sample_rate, fmt = _get_audio_info_from_bytes(audio_data, info.format)
            info.duration = duration
            info.sample_rate = sample_rate
            if fmt:
                info.format = fmt
        return info

    else:
        # File path
        info.source_type = 'file'
        duration, sample_rate, fmt = _get_audio_info_from_file(audio_source)
        info.duration = duration
        info.sample_rate = sample_rate
        if fmt:
            info.format = fmt
        elif not info.format:
            _, ext = os.path.splitext(audio_source)
            if ext:
                info.format = ext[1:].lower()

        return info


def get_video_info(video_source: str, video_format: Optional[str] = None) -> VideoInfo:
    """
    Extract information from a video source.

    Note: Full video metadata extraction requires additional dependencies.
    Currently extracts format from file extension or data URI.

    Args:
        video_source: Video URL, file path, or base64 data URI.
        video_format: Known video format (e.g., 'mp4', 'mov').

    Returns:
        VideoInfo with extracted metadata.
    """
    info = VideoInfo()
    if video_format:
        info.format = video_format.lower()

    if _is_http_url(video_source):
        info.source_type = 'url'
        if not info.format:
            ext_match = re.search(r'\.([a-zA-Z0-9]+)(?:\?|$)', video_source)
            if ext_match:
                info.format = ext_match.group(1).lower()

    elif _is_base64_data_uri(video_source):
        info.source_type = 'base64'
        if not info.format:
            info.format = _extract_format_from_data_uri(video_source)

    else:
        # File path
        info.source_type = 'file'
        if not info.format:
            _, ext = os.path.splitext(video_source)
            if ext:
                info.format = ext[1:].lower()

    return info


def extract_multimodal_content_from_sample(
    sample: 'Sample',
) -> Tuple[List[ImageInfo], List[AudioInfo], List[VideoInfo]]:
    """
    Extract all multimodal content from a sample.

    Args:
        sample: The Sample object to analyze.

    Returns:
        Tuple of (image_infos, audio_infos, video_infos).
    """
    images: List[ImageInfo] = []
    audios: List[AudioInfo] = []
    videos: List[VideoInfo] = []

    # Get input content
    input_content = sample.input
    if input_content is None:
        return images, audios, videos

    # Handle different input types
    if isinstance(input_content, str):
        return images, audios, videos

    # Process list of messages
    if isinstance(input_content, list):
        for item in input_content:
            # Check against ChatMessageBase (the actual base class)
            # ChatMessage is a Union type alias, not a class
            if isinstance(item, ChatMessageBase):
                content = item.content
                if isinstance(content, list):
                    for content_item in content:
                        if isinstance(content_item, ContentImage):
                            images.append(get_image_info(content_item.image))
                        elif isinstance(content_item, ContentAudio):
                            audios.append(get_audio_info(content_item.audio, getattr(content_item, 'format', None)))
                        elif isinstance(content_item, ContentVideo):
                            videos.append(get_video_info(content_item.video, getattr(content_item, 'format', None)))
            elif isinstance(item, ContentImage):
                images.append(get_image_info(item.image))
            elif isinstance(item, ContentAudio):
                audios.append(get_audio_info(item.audio, getattr(item, 'format', None)))
            elif isinstance(item, ContentVideo):
                videos.append(get_video_info(item.video, getattr(item, 'format', None)))

    return images, audios, videos


def compute_multimodal_statistics(
    all_image_infos: List[List[ImageInfo]],
    all_audio_infos: List[List[AudioInfo]],
    all_video_infos: List[List[VideoInfo]],
) -> MultimodalStatistics:
    """
    Compute aggregated multimodal statistics from collected info lists.

    Args:
        all_image_infos: List of image info lists, one per sample.
        all_audio_infos: List of audio info lists, one per sample.
        all_video_infos: List of video info lists, one per sample.

    Returns:
        MultimodalStatistics with aggregated metrics.
    """
    result = MultimodalStatistics()

    # Compute image statistics
    image_counts = [len(imgs) for imgs in all_image_infos]
    if any(count > 0 for count in image_counts):
        result.has_images = True
        all_images = [img for imgs in all_image_infos for img in imgs]

        # Collect unique formats
        formats: Set[str] = set()
        for img in all_images:
            if img.format:
                formats.add(img.format)

        # Collect resolutions and find min/max
        resolutions: Set[str] = set()
        resolution_pixels: List[Tuple[int, str]] = []
        for img in all_images:
            if img.width and img.height:
                res_str = f'{img.width}x{img.height}'
                resolutions.add(res_str)
                resolution_pixels.append((img.width * img.height, res_str))

        resolution_min = None
        resolution_max = None
        if resolution_pixels:
            resolution_pixels.sort(key=lambda x: x[0])
            resolution_min = resolution_pixels[0][1]
            resolution_max = resolution_pixels[-1][1]

        # Filter out zero counts for samples without images
        non_zero_counts = [c for c in image_counts if c > 0]
        result.image_stats = ImageStatistics(
            count_total=sum(image_counts),
            count_per_sample_min=min(non_zero_counts) if non_zero_counts else 0,
            count_per_sample_max=max(non_zero_counts) if non_zero_counts else 0,
            count_per_sample_mean=(stats_module.mean(non_zero_counts) if non_zero_counts else 0.0),
            resolutions=sorted(list(resolutions)),
            resolution_min=resolution_min,
            resolution_max=resolution_max,
            formats=sorted(list(formats)),
        )

    # Compute audio statistics
    audio_counts = [len(auds) for auds in all_audio_infos]
    if any(count > 0 for count in audio_counts):
        result.has_audio = True
        all_audios = [aud for auds in all_audio_infos for aud in auds]

        # Collect unique formats and sample rates
        formats: Set[str] = set()
        sample_rates: Set[int] = set()
        durations: List[float] = []

        for aud in all_audios:
            if aud.format:
                formats.add(aud.format)
            if aud.sample_rate:
                sample_rates.add(aud.sample_rate)
            if aud.duration is not None:
                durations.append(aud.duration)

        non_zero_counts = [c for c in audio_counts if c > 0]
        result.audio_stats = AudioStatistics(
            count_total=sum(audio_counts),
            count_per_sample_min=min(non_zero_counts) if non_zero_counts else 0,
            count_per_sample_max=max(non_zero_counts) if non_zero_counts else 0,
            count_per_sample_mean=(stats_module.mean(non_zero_counts) if non_zero_counts else 0.0),
            duration_min=min(durations) if durations else None,
            duration_max=max(durations) if durations else None,
            duration_mean=stats_module.mean(durations) if durations else None,
            sample_rates=sorted(list(sample_rates)),
            formats=sorted(list(formats)),
        )

    # Compute video statistics
    video_counts = [len(vids) for vids in all_video_infos]
    if any(count > 0 for count in video_counts):
        result.has_video = True
        all_videos = [vid for vids in all_video_infos for vid in vids]

        # Collect unique formats and resolutions
        formats: Set[str] = set()
        resolutions: Set[str] = set()
        durations: List[float] = []

        for vid in all_videos:
            if vid.format:
                formats.add(vid.format)
            if vid.width and vid.height:
                resolutions.add(f'{vid.width}x{vid.height}')
            if vid.duration is not None:
                durations.append(vid.duration)

        non_zero_counts = [c for c in video_counts if c > 0]
        result.video_stats = VideoStatistics(
            count_total=sum(video_counts),
            count_per_sample_min=min(non_zero_counts) if non_zero_counts else 0,
            count_per_sample_max=max(non_zero_counts) if non_zero_counts else 0,
            count_per_sample_mean=(stats_module.mean(non_zero_counts) if non_zero_counts else 0.0),
            duration_min=min(durations) if durations else None,
            duration_max=max(durations) if durations else None,
            duration_mean=stats_module.mean(durations) if durations else None,
            resolutions=sorted(list(resolutions)),
            formats=sorted(list(formats)),
        )

    return result


def compute_text_length(text) -> int:
    """
    Compute the character length of text content.

    Args:
        text: String or list of ChatMessage objects.

    Returns:
        Total character count.
    """
    if isinstance(text, str):
        return len(text)
    elif isinstance(text, list):
        # For chat messages, concatenate all content
        total_length = 0
        for msg in text:
            if isinstance(msg, ChatMessage):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if isinstance(content, str):
                    total_length += len(content)
                elif isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if isinstance(item, str):
                            total_length += len(item)
                        elif hasattr(item, 'text'):
                            total_length += len(item.text)
            else:
                total_length += len(str(msg))
        return total_length
    return len(str(text))


def compute_sample_lengths(sample: 'Sample') -> Tuple[int, int]:
    """
    Compute prompt and target lengths for a sample.

    Args:
        sample: Sample object.

    Returns:
        Tuple of (prompt_length, target_length).
    """
    prompt_length = compute_text_length(sample.input)

    # Handle target
    if isinstance(sample.target, list):
        target_length = sum(len(str(t)) for t in sample.target)
    else:
        target_length = len(str(sample.target)) if sample.target else 0

    return prompt_length, target_length


def compute_benchmark_statistics(
    adapter: 'DataAdapter',
    max_samples_per_subset: Optional[int] = None,
    compute_target_stats: bool = True,
    compute_multimodal: bool = True,
) -> 'DataStatistics':
    """
    Compute comprehensive statistics for a benchmark dataset.

    This function loads the benchmark dataset and computes various statistics
    including sample counts, prompt lengths, and optionally target lengths.
    For multimodal datasets, also computes image/audio/video statistics.

    Args:
        adapter: The DataAdapter instance for the benchmark.
        max_samples_per_subset: Maximum samples to analyze per subset (for large datasets).
                               None means analyze all samples.
        compute_target_stats: Whether to compute target/answer length statistics.
        compute_multimodal: Whether to compute multimodal statistics (images, audio, video).

    Returns:
        DataStatistics object with computed metrics.

    Example:
        >>> from evalscope.api.registry import get_benchmark
        >>> adapter = get_benchmark('gsm8k')
        >>> stats = compute_benchmark_statistics(adapter)
        >>> print(f"Total samples: {stats.total_samples}")
    """
    logger.info(f'Computing statistics for benchmark: {adapter.name}')

    # Ensure adapter has necessary configuration for loading
    # Create a minimal task config if not present
    if adapter._task_config is None:
        try:
            from evalscope.config import TaskConfig
            adapter._task_config = TaskConfig(model='dummy', datasets=[adapter.name])
        except Exception as e:
            logger.warning(f'Could not create TaskConfig for {adapter.name}: {e}')
            return DataStatistics(computed_at=datetime.now().isoformat())

    try:
        test_dataset = adapter.load_dataset()
    except Exception as e:
        logger.warning(f'Failed to load dataset for {adapter.name}: {e}')
        return DataStatistics(computed_at=datetime.now().isoformat())

    all_prompt_lengths: List[int] = []
    all_target_lengths: List[int] = []
    subset_stats_list: List[SubsetStatistics] = []

    # For multimodal statistics aggregation
    all_image_infos: List[List[ImageInfo]] = []
    all_audio_infos: List[List[AudioInfo]] = []
    all_video_infos: List[List[VideoInfo]] = []

    for subset_name, dataset in test_dataset.items():
        samples = list(dataset)

        # Apply sample limit if specified
        if max_samples_per_subset and len(samples) > max_samples_per_subset:
            samples_to_analyze = samples[:max_samples_per_subset]
        else:
            samples_to_analyze = samples

        prompt_lengths: List[int] = []
        target_lengths: List[int] = []

        # Per-subset multimodal info
        subset_image_infos: List[List[ImageInfo]] = []
        subset_audio_infos: List[List[AudioInfo]] = []
        subset_video_infos: List[List[VideoInfo]] = []

        for sample in samples_to_analyze:
            p_len, t_len = compute_sample_lengths(sample)
            prompt_lengths.append(p_len)
            if compute_target_stats:
                target_lengths.append(t_len)

            # Extract multimodal content
            if compute_multimodal:
                images, audios, videos = extract_multimodal_content_from_sample(sample)
                subset_image_infos.append(images)
                subset_audio_infos.append(audios)
                subset_video_infos.append(videos)

        all_prompt_lengths.extend(prompt_lengths)
        all_target_lengths.extend(target_lengths)
        all_image_infos.extend(subset_image_infos)
        all_audio_infos.extend(subset_audio_infos)
        all_video_infos.extend(subset_video_infos)

        # Compute subset statistics
        if prompt_lengths:
            # Compute subset multimodal stats if applicable
            subset_multimodal = None
            if compute_multimodal and (subset_image_infos or subset_audio_infos or subset_video_infos):
                subset_multimodal = compute_multimodal_statistics(
                    subset_image_infos, subset_audio_infos, subset_video_infos
                )
                # Only include if there's actual multimodal content
                if not (subset_multimodal.has_images or subset_multimodal.has_audio or subset_multimodal.has_video):
                    subset_multimodal = None

            subset_stat = SubsetStatistics(
                name=subset_name,
                sample_count=len(dataset),  # Use original count, not limited
                prompt_length_mean=stats_module.mean(prompt_lengths),
                prompt_length_min=min(prompt_lengths),
                prompt_length_max=max(prompt_lengths),
                prompt_length_std=stats_module.stdev(prompt_lengths) if len(prompt_lengths) > 1 else 0.0,
                target_length_mean=stats_module.mean(target_lengths) if target_lengths else None,
                multimodal=subset_multimodal,
            )
            subset_stats_list.append(subset_stat)

    # Compute overall statistics
    total_samples = sum(len(dataset) for dataset in test_dataset.values())

    # Compute overall multimodal stats
    overall_multimodal = None
    if compute_multimodal and (all_image_infos or all_audio_infos or all_video_infos):
        overall_multimodal = compute_multimodal_statistics(all_image_infos, all_audio_infos, all_video_infos)
        # Only include if there's actual multimodal content
        if not (overall_multimodal.has_images or overall_multimodal.has_audio or overall_multimodal.has_video):
            overall_multimodal = None

    result = DataStatistics(
        total_samples=total_samples,
        subset_stats=subset_stats_list,
        prompt_length_mean=stats_module.mean(all_prompt_lengths) if all_prompt_lengths else 0.0,
        prompt_length_min=min(all_prompt_lengths) if all_prompt_lengths else 0,
        prompt_length_max=max(all_prompt_lengths) if all_prompt_lengths else 0,
        prompt_length_std=stats_module.stdev(all_prompt_lengths) if len(all_prompt_lengths) > 1 else 0.0,
        target_length_mean=stats_module.mean(all_target_lengths) if all_target_lengths else None,
        multimodal=overall_multimodal,
        computed_at=datetime.now().isoformat(),
    )

    # Log summary
    multimodal_info = ''
    if overall_multimodal:
        parts = []
        if overall_multimodal.has_images and overall_multimodal.image_stats:
            parts.append(f'{overall_multimodal.image_stats.count_total} images')
        if overall_multimodal.has_audio and overall_multimodal.audio_stats:
            parts.append(f'{overall_multimodal.audio_stats.count_total} audio files')
        if overall_multimodal.has_video and overall_multimodal.video_stats:
            parts.append(f'{overall_multimodal.video_stats.count_total} videos')
        if parts:
            multimodal_info = f', multimodal: {", ".join(parts)}'

    logger.info(
        f'Statistics computed: {total_samples} total samples, '
        f'{len(subset_stats_list)} subsets, '
        f'mean prompt length: {result.prompt_length_mean:.1f}{multimodal_info}'
    )

    return result


def get_sample_example(
    adapter: 'DataAdapter',
    subset: Optional[str] = None,
    sample_index: int = 0,
    max_length: int = 500,
) -> Optional['SampleExample']:
    """
    Extract a representative sample example from the benchmark.

    Args:
        adapter: The DataAdapter instance for the benchmark.
        subset: Specific subset to get example from. If None, uses first available.
        sample_index: Index of sample to extract (default 0 for first sample).
        max_length: Maximum length for string values before truncation.

    Returns:
        SampleExample object with the extracted sample, or None if no samples available.

    Example:
        >>> from evalscope.api.registry import get_benchmark
        >>> adapter = get_benchmark('gsm8k')
        >>> example = get_sample_example(adapter)
        >>> print(example.to_json_block())
    """
    from evalscope.api.benchmark.statistics import SampleExample

    logger.info(f'Extracting sample example from benchmark: {adapter.name}')

    # Ensure adapter has necessary configuration for loading
    if adapter._task_config is None:
        try:
            from evalscope.config import TaskConfig
            adapter._task_config = TaskConfig(model='dummy', datasets=[adapter.name])
        except Exception as e:
            logger.warning(f'Could not create TaskConfig for {adapter.name}: {e}')
            return None

    try:
        test_dataset = adapter.load_dataset()
    except Exception as e:
        logger.warning(f'Failed to load dataset for {adapter.name}: {e}')
        return None

    if not test_dataset:
        logger.warning(f'No dataset loaded for {adapter.name}')
        return None

    # Determine target subset
    if subset and subset in test_dataset.keys():
        target_subset = subset
    else:
        target_subset = next(iter(test_dataset.keys()))

    dataset = test_dataset.get(target_subset)
    if not dataset or len(dataset) == 0:
        logger.warning(f'No samples in subset {target_subset} for {adapter.name}')
        return None

    samples = list(dataset)
    if sample_index >= len(samples):
        sample_index = 0

    sample = samples[sample_index]

    return SampleExample.from_sample(
        sample,
        subset=target_subset,
        max_length=max_length,
    )


if __name__ == '__main__':
    # Example usage
    from evalscope.api.registry import get_benchmark
    adapter = get_benchmark('gsm8k')
    stats = compute_benchmark_statistics(adapter)
