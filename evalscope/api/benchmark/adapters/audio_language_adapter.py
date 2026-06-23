"""AudioLanguageAdapter: base class for audio-language benchmarks.

Provides shared utilities for benchmarks that process audio inputs
(ASR, audio QA, spoken language understanding, etc.). Parallel to
:class:`VisionLanguageAdapter` which handles image/video modalities.
"""

from .default_data_adapter import DefaultDataAdapter


class AudioLanguageAdapter(DefaultDataAdapter):
    """Adapter for audio-language benchmarks (ASR, audio QA, etc.).

    Subclasses typically construct :class:`ContentAudio` objects in
    ``record_to_sample`` and may use the :meth:`_to_wav` helper to
    normalize raw audio bytes into WAV format before base64-encoding.
    """

    @staticmethod
    def _to_wav(raw_bytes: bytes) -> bytes:
        """Convert audio bytes (OPUS/OGG/FLAC/etc.) to WAV format using soundfile.

        Args:
            raw_bytes: Raw audio bytes in any format supported by libsndfile.

        Returns:
            WAV-encoded bytes.
        """
        import io

        from evalscope.utils.import_utils import check_import
        check_import('soundfile', raise_error=True, feature_name='AudioLanguageAdapter')
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(raw_bytes))
        wav_buf = io.BytesIO()
        sf.write(wav_buf, data, sr, format='WAV')
        return wav_buf.getvalue()
