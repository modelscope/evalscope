# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared helpers for the AIR-Bench Foundation/Chat adapters."""

import json
import os
import random
import shutil
import subprocess
from copy import deepcopy
from hashlib import sha1
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.utils.logger import get_logger

logger = get_logger()

HF_REPO_ID = 'evalscope/AIR-Bench'

# Mapping from Foundation track sub-directory name -> high-level audio category.
# The 25 directory names correspond to the 19 logical tasks described in the
# AIR-Bench paper (gender / emotion / acoustic scene / sound-QA / music genre /
# music instruments tasks each draw from two source datasets).
FOUNDATION_SUBSET_TO_CATEGORY: Dict[str, str] = {
    # ---- speech (11 dirs / 9 tasks) ----
    'Speaker_Age_Prediction_common_voice_13.0_en': 'speech',
    'Speaker_Emotion_Recontion_iemocap': 'speech',
    'Speaker_Emotion_Recontion_meld': 'speech',
    'Speaker_Gender_Recognition_common_voice_13_en': 'speech',
    'Speaker_Gender_Recognition_meld': 'speech',
    'Speaker_Intent_Classification_slurp': 'speech',
    'Speaker_Number_Verification_voxceleb1': 'speech',
    'Speech_Entity_Reconition_slurp': 'speech',
    'Speech_Grounding_librispeech': 'speech',
    'Spoken_Language_Identification_covost2': 'speech',
    'Synthesized_Voice_Detection_fake_or_real': 'speech',
    # ---- sound (6 dirs / 4 tasks) ----
    'Acoustic_Scene_Classification_CochlScene': 'sound',
    'Acoustic_Scene_Classification_TUT2017': 'sound',
    'Audio_Grounding_AudioGrounding': 'sound',
    'Sound_AQA_avqa': 'sound',
    'Sound_AQA_clothoaqa': 'sound',
    'vocal_sound_classification_VocalSound': 'sound',
    # ---- music (8 dirs / 6 tasks) ----
    'Music_AQA_music_avqa': 'music',
    'Music_Genre_Recognition_MTJ-Jamendo': 'music',
    'Music_Genre_Recognition_fma': 'music',
    'Music_Instruments_Classfication_MTJ-Jamendo': 'music',
    'Music_Instruments_Classfication_nsynth': 'music',
    'Music_Midi_Pitch_Analysis_nsynth': 'music',
    'Music_Midi_Velocity_Analysis_nsynth': 'music',
    'Music_Mood_Recognition_MTJ-Jamendo': 'music',
}

FOUNDATION_REPORTED_CATEGORIES = ['speech', 'sound', 'music']

# Mapping from Chat track task_name -> aggregation category.
# Per the official cal_score.py logic.
CHAT_TASK_TO_CATEGORY: Dict[str, str] = {
    'speech_QA': 'speech',
    'speech_dialogue_QA': 'speech',
    'sound_QA': 'sound',
    'sound_generation_QA': 'sound',
    'music_QA': 'music',
    'music_generation_analysis_QA': 'music',
    'speech_and_sound_QA': 'speech_and_sound',
    'speech_and_music_QA': 'speech_and_music',
}

CHAT_REPORTED_CATEGORIES = ['speech', 'sound', 'music', 'speech_and_sound', 'speech_and_music']

SUPPORTED_CONTENT_AUDIO_FORMATS = {'wav', 'mp3'}


def _resolve_local_cache(
    dataset_id: str,
    track: str,
    cache_dir: Optional[str],
    subset_dirs: Optional[List[str]],
) -> Optional[str]:
    """Return the local cache root if all requested subset directories are present.

    Checks the standard ModelScope cache layout
    ``{cache_dir}/{dataset_id}/{track}/{subset}/`` for every entry in
    *subset_dirs*.  When *subset_dirs* is ``None`` the check only verifies
    that the track-level directory exists.

    Returns:
        The dataset root path (parent of the track directory) when the cache
        is complete for the requested subsets, or ``None`` if any subset is
        absent and a download is required.
    """
    cache_root = cache_dir or os.path.join(os.path.expanduser('~'), '.cache', 'modelscope', 'hub', 'datasets')
    local_cached = os.path.join(cache_root, *dataset_id.split('/'))
    track_dir = os.path.join(local_cached, track)

    if not os.path.isdir(track_dir):
        return None

    if subset_dirs:
        missing = [s for s in subset_dirs if not os.path.isdir(os.path.join(track_dir, s))]
        if missing:
            logger.info(f'AIR-Bench {track}: {len(missing)} subset(s) absent from local cache: {missing}.')
            return None

    return local_cached


def download_air_bench(
    track: str,
    dataset_id: str = HF_REPO_ID,
    cache_dir: Optional[str] = None,
    subset_dirs: Optional[List[str]] = None,
) -> str:
    """Resolve the AIR-Bench dataset root, downloading from ModelScope if needed.

    Args:
        track: Either ``'Foundation'`` or ``'Chat'`` — used to scope the download.
        dataset_id: Local path or ModelScope repository id. If a local path
            exists it is returned as-is.
        cache_dir: Optional local cache directory; falls back to the
            ``modelscope`` default when ``None``.
        subset_dirs: When provided, only these sub-directories under the chosen
            track are downloaded (alongside the track-level meta JSON).

    Returns:
        Absolute path to the dataset root containing ``Foundation/`` and/or
        ``Chat/`` sub-directories.
    """
    if os.path.isdir(dataset_id):
        return dataset_id

    # Fast path: return immediately when every requested subset is cached.
    local_cached = _resolve_local_cache(dataset_id, track, cache_dir, subset_dirs)
    if local_cached is not None:
        logger.info(
            f'AIR-Bench {track} already present in local cache at `{local_cached}`. '
            'Skipping remote download.'
        )
        return local_cached

    from modelscope import dataset_snapshot_download

    # Determine which subsets still need to be fetched.
    cache_root = cache_dir or os.path.join(os.path.expanduser('~'), '.cache', 'modelscope', 'hub', 'datasets')
    local_cached = os.path.join(cache_root, *dataset_id.split('/'))
    track_dir = os.path.join(local_cached, track)

    if os.path.isdir(track_dir) and subset_dirs:
        # Partial cache: only pull the absent subsets.
        missing = [s for s in subset_dirs if not os.path.isdir(os.path.join(track_dir, s))]
    else:
        missing = subset_dirs or []

    # Build allow-patterns: always include the meta JSON.
    allow_patterns: List[str] = [f'{track}/{track}_meta.json']
    if missing:
        for sub in missing:
            allow_patterns.append(f'{track}/{sub}/*')
    elif subset_dirs:
        for sub in subset_dirs:
            allow_patterns.append(f'{track}/{sub}/*')
    else:
        allow_patterns.append(f'{track}/*')

    logger.info(
        f'Downloading AIR-Bench {track} from ModelScope (`{dataset_id}`). '
        f'Patterns: {allow_patterns}. This can be tens of GB on first run.'
    )
    local_path = dataset_snapshot_download(
        dataset_id,
        cache_dir=cache_dir,
        allow_file_pattern=allow_patterns,
    )
    return local_path


def load_meta(track_root: str, track: str) -> List[Dict[str, Any]]:
    """Load the ``{track}_meta.json`` file from the dataset root."""
    meta_path = os.path.join(track_root, track, f'{track}_meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'AIR-Bench meta file not found at {meta_path}. '
            'Make sure the dataset has been downloaded correctly.'
        )
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_audio_path(
    track_root: str,
    track: str,
    task_name: str,
    dataset_name: str,
    rel_path: str,
) -> Tuple[str, str]:
    """Resolve a meta entry into a concrete audio file path and format string.

    Returns ``(absolute_path, audio_format)``. ``audio_format`` is the source
    file extension and may need normalisation before being passed to
    ``ContentAudio``.

    The Audio_Grounding task ships ``.flac`` files even though the meta JSON
    records ``.wav`` paths — we mirror the official inference script's
    extension-swap behaviour.
    """
    folder = f'{task_name}_{dataset_name}'
    if task_name == 'Audio_Grounding' and rel_path.lower().endswith('.wav'):
        rel_path = rel_path[:-4] + '.flac'
    audio_path = os.path.join(track_root, track, folder, rel_path)

    ext = os.path.splitext(rel_path)[1].lower().lstrip('.')
    audio_format = ext or 'wav'
    return audio_path, audio_format


def prepare_samples(
    samples: List[Sample],
    *,
    limit: Optional[int | float],
    repeats: int,
    shuffle: bool,
    seed: Optional[int],
    name: str,
) -> MemoryDataset:
    """Apply EvalScope dataset semantics to a hand-built sample list.

    AIR-Bench is not loaded through the generic ``DataLoader``, so this helper
    mirrors the important parts of that loader: shuffle, limit, repeat, and
    stable sample/group ids for caching.
    """
    prepared = list(samples)
    if shuffle:
        random.Random(seed).shuffle(prepared)

    if limit is not None:
        if isinstance(limit, float):
            limit = int(len(prepared) * limit)
        prepared = prepared[:limit]

    if repeats > 1:
        prepared = [deepcopy(sample) for sample in prepared for _ in range(repeats)]

    dataset = MemoryDataset(samples=prepared, name=name)
    dataset.reindex(group_size=repeats)
    return dataset


def normalise_audio_for_content(
    audio_path: str,
    audio_format: str,
    *,
    cache_dir: str,
) -> Tuple[str, str]:
    """Return an audio path/format pair accepted by ``ContentAudio``.

    EvalScope's OpenAI-compatible path currently accepts ``wav`` and ``mp3``.
    AIR-Bench includes FLAC files, so those are transcoded lazily into a cache
    directory. The original dataset snapshot is never modified.
    """
    audio_format = audio_format.lower()
    if audio_format in SUPPORTED_CONTENT_AUDIO_FORMATS:
        return audio_path, audio_format
    if audio_format != 'flac':
        raise ValueError(
            f'Unsupported AIR-Bench audio format: {audio_format!r} for {audio_path}. '
            f'Supported formats are {sorted(SUPPORTED_CONTENT_AUDIO_FORMATS)} and flac.'
        )

    return _convert_flac_to_wav(audio_path, cache_dir=cache_dir), 'wav'


def _convert_flac_to_wav(audio_path: str, *, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _file_sha1(audio_path)
    wav_path = os.path.join(cache_dir, f'{cache_key}.wav')
    if os.path.exists(wav_path):
        return wav_path

    try:
        import soundfile as sf

        data, sample_rate = sf.read(audio_path, dtype='float32', always_2d=False)
        sf.write(wav_path, data, sample_rate, format='WAV')
        return wav_path
    except ImportError:
        pass
    except Exception as e:
        raise RuntimeError(f'Failed to convert AIR-Bench FLAC audio with soundfile: {audio_path}') from e

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        try:
            subprocess.run(
                [ffmpeg, '-y', '-loglevel', 'error', '-i', audio_path, wav_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return wav_path
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
            raise RuntimeError(f'Failed to convert AIR-Bench FLAC audio with ffmpeg: {stderr}') from e

    raise ImportError(
        'AIR-Bench contains FLAC audio, but neither `soundfile` nor `ffmpeg` is available '
        'to convert it to WAV for model input. Install `evalscope[air_bench]`, '
        '`pip install soundfile`, or make `ffmpeg` available on PATH.'
    )


def audio_path_to_base64(audio_path: str, audio_format: str) -> str:
    """Read a local audio file and return a base64-encoded data-URI string.

    This mirrors the approach used by ``LibriSpeechAdapter``: the audio bytes
    are encoded as ``data:audio/<format>;base64,...`` so the content can be
    forwarded directly to an OpenAI-compatible API without relying on the
    server being able to access a local file path.
    """
    from evalscope.utils.io_utils import bytes_to_base64

    with open(audio_path, 'rb') as fh:
        return bytes_to_base64(fh.read(), format=audio_format, add_header=True, content_type='audio')


def _file_sha1(path: str) -> str:
    digest = sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
