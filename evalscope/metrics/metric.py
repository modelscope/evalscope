import base64
import binascii
import json
import math
import os
import requests
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.metric import Aggregator, AggScore, Metric, SampleScore, SingletonMetric, T2IMetric
from evalscope.api.registry import register_aggregation, register_metric
from evalscope.utils.import_utils import check_import
from .metrics import calculate_pass_at_k, calculate_pass_hat_k, mean, normalize_text

# ##################
# NLP Metrics ######
# ##################


@register_metric(name='exact_match')
class ExactMatch(Metric):

    def apply(self, predictions, references):
        return [
            float(normalize_text(prediction) == normalize_text(reference))
            for prediction, reference in zip(predictions, references)
        ]


@register_metric(name='acc')
class Accuracy(ExactMatch):

    def __init__(self, allow_inclusion: bool = False, numeric: bool = False):
        self.allow_inclusion = allow_inclusion
        self.numeric = numeric

    def apply(self, predictions, references):
        if self.allow_inclusion:
            results = []
            for prediction, reference in zip(predictions, references):
                if prediction and prediction in reference:
                    results.append(1.0)
                else:
                    results.append(0.0)
            return results
        elif self.numeric:
            from .math_parser import math_equal, strip_answer_string

            results = []
            for prediction, reference in zip(predictions, references):
                ref_answer = strip_answer_string(reference)
                results.append(float(math_equal(prediction, ref_answer)))

            return results
        else:
            return super().apply(predictions, references)


@register_metric(name='numeric_match')
class NumericMatch(Metric):

    def apply(self, predictions, references):
        return [float(prediction == reference) for prediction, reference in zip(predictions, references)]


@register_metric(name='math_acc')
class MathAcc(Metric):

    def apply(self, predictions, references):
        from .math_parser import extract_answer, math_equal, strip_answer_string

        results = []
        for prediction, reference in zip(predictions, references):
            pred_answer = strip_answer_string(extract_answer(prediction))
            ref_answer = strip_answer_string(reference)
            results.append(float(math_equal(pred_answer, ref_answer)))

        return results


@register_metric(name='multi_choice_acc')
class MultiChoiceAcc(Metric):

    def apply(self, predictions, references):
        """
        Calculate accuracy for multiple-choice questions.

        Args:
            predictions (List[str]): List of predicted answers.
            references (List[str]): List of correct answers.

        Returns:
            List[float]: List of accuracy scores (1.0 for correct, 0.0 for incorrect).
        """
        res = []
        for prediction, reference in zip(predictions, references):
            prediction = set(prediction.strip().upper())
            reference = set(reference.strip().upper())
            # if the prediction has answer that not in reference, it is wrong
            if not prediction.issubset(reference):
                res.append(0.0)
                continue
            common = prediction.intersection(reference)
            res.append(len(common) / len(reference) if reference else 0.0)
        return res


@register_metric(name='anls')
class ANLS(Metric):

    def __init__(self, thresh_hold=0.5):
        self.thresh_hold = thresh_hold

    def apply(self, predictions, references):
        """
        Calculate ANLS (Average Normalized Levenshtein Similarity) for a list of predictions and references.
        This implementation is adapted from
        https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py

        Args:
            references (List[str]): List of correct answers. Each answer can be a string of json.
            predictions (List[str]): List of predicted answers.
        """
        from .metrics import levenshtein_distance

        res = []
        # Unwrap predictions if it's a nested list
        for prediction, reference in zip(predictions, references):
            # Parse the reference which is a json string
            try:
                answer = json.loads(reference)
            except json.JSONDecodeError:
                answer = reference
            if isinstance(answer, str):
                answer = [answer]
            assert isinstance(answer, list), 'The reference answer should be a list of answers.'

            # Calculate ANLS for each reference answer
            values = []
            for ans in answer:
                # preprocess both the answers - gt and prediction
                gt_answer = ' '.join(ans.strip().lower().split())
                det_answer = ' '.join(prediction.strip().lower().split())

                dist = levenshtein_distance(gt_answer, det_answer)
                length = max(len(ans.upper()), len(prediction.upper()))
                values.append(0.0 if length == 0 else float(dist) / float(length))

            question_result = 0.0
            if values:
                question_result = 1 - min(values)
                if question_result < self.thresh_hold:
                    question_result = 0.0
            res.append(question_result)
        return res


@register_metric(name='wer')
class WER(Metric):

    def __init__(self, language: str = 'English'):
        self.language = language

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        from .text_normalizer.wer import wer

        return [wer([ref], [pred], self.language) for pred, ref in zip(predictions, references)]


@register_metric(name='audio_wer')
class AudioWER(Metric):
    """WER metric for generated audio using a remote ASR endpoint."""

    DEFAULT_OPENAI_API_BASE = 'https://api.openai.com/v1'
    DEFAULT_SEED_TTS_TRANSCRIPTIONS_PATH = '/audio/transcriptions'
    DEFAULT_SEED_TTS_RESPONSES_PATH = '/responses'
    SUPPORTED_AUDIO_PROTOCOLS = {'transcriptions', 'responses'}

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        language: str = 'en',
        api_protocol: Optional[str] = None,
        prompt: Optional[str] = None,
        timeout: float = 60,
    ):
        self.api_base = (
            api_base or os.getenv('SEED_TTS_EVAL_ASR_API_BASE') or os.getenv('OPENAI_BASE_URL')
            or self.DEFAULT_OPENAI_API_BASE
        ).rstrip('/')
        self.api_key = api_key or os.getenv('SEED_TTS_EVAL_ASR_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.model = model or os.getenv('SEED_TTS_EVAL_ASR_MODEL') or 'whisper-1'
        self.language = self._normalize_language(language)
        self.api_protocol = self._resolve_protocol(api_protocol)
        self.prompt = prompt or os.getenv('SEED_TTS_EVAL_ASR_PROMPT') or 'Transcribe the speech. Return only the text.'
        self.timeout = timeout
        self.transcriptions: List[str] = []
        self._session = requests.Session()

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        from .text_normalizer.wer import normalize_text, wer

        self.transcriptions = [self._transcribe(prediction) for prediction in predictions]
        scores = []
        for transcript, reference in zip(self.transcriptions, references):
            normalized_prediction = normalize_text(transcript, self.language)
            normalized_reference = normalize_text(reference, self.language)
            scores.append(wer([normalized_reference], [normalized_prediction], self.language))
        return scores

    def _transcribe(self, audio: str) -> str:
        if not self.api_key:
            raise ValueError('api_key is required for audio_wer. Set SEED_TTS_EVAL_ASR_API_KEY or OPENAI_API_KEY.')

        if self.api_protocol == 'responses':
            return self._transcribe_with_responses(audio)
        if self.api_protocol != 'transcriptions':
            raise ValueError(f'Unsupported audio_wer api_protocol: {self.api_protocol}')
        return self._transcribe_with_transcriptions(audio)

    def _transcribe_with_transcriptions(self, audio: str) -> str:
        from evalscope.utils.url_utils import file_as_data, is_data_uri

        endpoint = self._transcription_endpoint()
        audio_bytes, mime_type = file_as_data(audio, default_mime_type='audio/wav')
        filename = 'audio.wav' if is_data_uri(audio) else os.path.basename(audio.split('?', 1)[0]) or 'audio.wav'
        headers = {'Authorization': f'Bearer {self.api_key}'}
        data = {
            'model': self.model,
            'response_format': 'json',
        }
        if self.language:
            data['language'] = 'zh' if self.language == 'cmn_hans' else self.language

        response = self._session.post(
            endpoint,
            headers=headers,
            files={'file': (filename, audio_bytes, mime_type)},
            data=data,
            timeout=self.timeout,
        )
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError:
            return response.text.strip()
        if not isinstance(payload, dict):
            raise ValueError(f'Unexpected response payload format (expected dict): {payload}')

        text = payload.get('text') or payload.get('transcription')
        if text is None and isinstance(payload.get('result'), dict):
            text = payload['result'].get('text')
        if text is None:
            raise ValueError(f'No transcription text found in response: {payload}')
        return str(text)

    def _transcribe_with_responses(self, audio: str) -> str:
        response = self._session.post(
            self._responses_endpoint(),
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': self.model,
                'input': [{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_audio',
                            'audio_url': self._audio_url(audio),
                        },
                        {
                            'type': 'input_text',
                            'text': self.prompt,
                        },
                    ],
                }],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f'Unexpected response payload format (expected dict): {payload}')

        output_text = payload.get('output_text')
        if output_text:
            return str(output_text)

        text_parts: List[str] = []
        for item in payload.get('output', []):
            if item.get('type') != 'message':
                continue
            for content in item.get('content', []):
                if content.get('type') in {'output_text', 'text'} and content.get('text'):
                    text_parts.append(str(content['text']))
        if not text_parts:
            raise ValueError(f'No transcription text found in response: {payload}')
        return '\n'.join(text_parts).strip()

    def _transcription_endpoint(self) -> str:
        return self._build_api_endpoint(self.api_base, self.DEFAULT_SEED_TTS_TRANSCRIPTIONS_PATH)

    def _responses_endpoint(self) -> str:
        return self._build_api_endpoint(self.api_base, self.DEFAULT_SEED_TTS_RESPONSES_PATH)

    @staticmethod
    def _audio_url(audio: str) -> str:
        from evalscope.utils.url_utils import file_as_data_uri, is_data_uri, is_http_url

        if is_data_uri(audio) or is_http_url(audio):
            return audio
        return file_as_data_uri(audio, default_mime_type='audio/wav')

    @staticmethod
    def _normalize_language(language: str) -> str:
        language_map = {
            'zh': 'cmn_hans',
            'cn': 'cmn_hans',
            'cmn': 'cmn_hans',
            'chinese': 'cmn_hans',
            'en': 'en',
            'english': 'en',
        }
        return language_map.get((language or '').lower(), language)

    @staticmethod
    def _build_api_endpoint(api_base: str, suffix: str) -> str:
        if api_base.endswith(suffix):
            return api_base
        return f'{api_base.rstrip("/")}/{suffix.lstrip("/")}'

    def _resolve_protocol(self, api_protocol: Optional[str]) -> str:
        protocol = (api_protocol or os.getenv('SEED_TTS_EVAL_ASR_API_PROTOCOL') or 'transcriptions').lower()
        if protocol not in self.SUPPORTED_AUDIO_PROTOCOLS:
            raise ValueError(f'Unsupported audio_wer api_protocol: {protocol}')
        return protocol


@register_metric(name='bertscore')
class BertScore(SingletonMetric):

    def _init_once(self, model_id_or_path: str = 'google-bert/bert-base-chinese', **kwargs):
        """BertScore metric.

        Args:
            model_id_or_path (str, optional): The model ID on modelscope or path to the pre-trained model.
                Defaults to 'google-bert/bert-base-chinese'.
        """
        check_import('torch', 'torch', raise_error=True, feature_name='BertScore Metric')

        from .bert_score.scorer import BERTScorer
        self.scorer = BERTScorer(model_id_or_path=model_id_or_path, batch_size=1024, **kwargs)

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        _, _, F1 = self.scorer.score(predictions, references)
        return [round(f1.item(), 6) for f1 in F1]


@register_metric(name='comet')
class COMETScore(SingletonMetric):

    def _init_once(self, model_id_or_path: str = 'evalscope/wmt22-comet-da'):
        """COMETScore metric.

        Args:
            model_name (str, optional): The model name on huggingface.
                Defaults to 'evalscope/wmt22-comet-da'.
        """
        check_import('comet', 'unbabel-comet', raise_error=True, feature_name='COMETScore Metric')

        from comet import load_from_checkpoint
        from modelscope import snapshot_download

        self.model_name = model_id_or_path
        model_path = snapshot_download(model_id_or_path)
        checkpoint_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')
        self.comet_scorer = load_from_checkpoint(checkpoint_path)

    def apply(self, samples: List[Dict[str, str]]) -> List[float]:
        """Apply COMET scoring."""
        import torch

        model_output = self.comet_scorer.predict(
            samples=samples,
            batch_size=1024,
            gpus=1 if torch.cuda.is_available() else 0,
            progress_bar=False,
        )
        scores = model_output.scores if hasattr(model_output, 'scores') else [model_output.system_score] * len(samples)

        return [round(score, 6) for score in scores]


@register_metric(name='cer')
class CER(Metric):

    def __init__(self, language: str = 'en'):
        self.language = language

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        from jiwer import cer as jiwer_cer

        from evalscope.metrics.text_normalizer.wer import normalize_text

        return [
            jiwer_cer(normalize_text(ref, self.language), normalize_text(pred, self.language))
            for pred, ref in zip(predictions, references)
        ]


@register_metric(name='sem_score')
class SemScore(SingletonMetric):

    def _init_once(self, **kwargs):
        """SemScore metric.
        """
        check_import('bert_score', 'bert-score', raise_error=True, feature_name='SemScore Metric')
        check_import('torch', 'torch', raise_error=True, feature_name='SemScore Metric')

        from .sem_score.scorer import SemScorer
        self.scorer = SemScorer(batch_size=1024, **kwargs)

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        scores = self.scorer.score_all(predictions, references)
        return [round(score, 6) for score in scores]


# #######################
# Image Pair Metrics ####
# #######################


class ImagePairMixin:
    """Utilities for metrics that compare a generated image with a reference image."""

    image_pair_metric = True

    def _prepare_pil_images(self, prediction: Any, reference: Any) -> Tuple[Any, Any]:
        prediction_image = self._load_pil_image(prediction)
        reference_image = self._load_pil_image(reference)
        if prediction_image.size == reference_image.size:
            return prediction_image, reference_image
        if not self.resize:
            raise ValueError(
                f'Image sizes do not match: prediction={prediction_image.size}, reference={reference_image.size}. '
                'Set resize=True to resize the reference image before scoring.'
            )

        from PIL import Image

        resample = getattr(getattr(Image, 'Resampling', Image), 'BICUBIC')
        reference_image = reference_image.resize(prediction_image.size, resample=resample)
        return prediction_image, reference_image

    def _prepare_arrays(self, prediction: Any, reference: Any) -> Tuple[Any, Any]:
        prediction_image, reference_image = self._prepare_pil_images(prediction, reference)
        return self._image_to_array(prediction_image), self._image_to_array(reference_image)

    @staticmethod
    def _load_pil_image(image: Any) -> Any:
        from io import BytesIO
        from PIL import Image

        from evalscope.utils.url_utils import file_as_data

        if image is None or (isinstance(image, str) and image.strip() == ''):
            raise ValueError('Image value is required for image pair metrics.')
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        if isinstance(image, bytes):
            return Image.open(BytesIO(image)).convert('RGB')
        if isinstance(image, dict):
            for key in ('bytes', 'image', 'image_path', 'path', 'url'):
                if key in image and image[key] is not None:
                    return ImagePairMixin._load_pil_image(image[key])
            raise ValueError(f'No image value found in dict keys: {sorted(image.keys())}')
        if hasattr(image, 'shape'):
            return ImagePairMixin._array_to_pil_image(image)
        if not isinstance(image, str):
            raise TypeError(f'Unsupported image value type: {type(image)}')

        image = image.strip()
        try:
            image_bytes, _ = file_as_data(image)
        except OSError as error:
            try:
                image_bytes = base64.b64decode(''.join(image.split()), validate=True)
            except (binascii.Error, ValueError):
                raise error

        return Image.open(BytesIO(image_bytes)).convert('RGB')

    @staticmethod
    def _array_to_pil_image(image: Any) -> Any:
        check_import('numpy', 'numpy', raise_error=True, feature_name='Image pair metrics')
        import numpy as np
        from PIL import Image

        array = np.asarray(image)
        if array.ndim == 2:
            pass
        elif array.ndim == 3 and array.shape[2] in (1, 3, 4):
            if array.shape[2] == 1:
                array = array[:, :, 0]
        else:
            raise ValueError(f'Unsupported image array shape: {array.shape}')
        if np.issubdtype(array.dtype, np.floating):
            array = array * 255 if array.size and array.max() <= 1.0 else array
        array = np.clip(array, 0, 255).astype('uint8')
        return Image.fromarray(array).convert('RGB')

    @staticmethod
    def _image_to_array(image: Any) -> Any:
        check_import('numpy', 'numpy', raise_error=True, feature_name='Image pair metrics')
        import numpy as np

        return np.asarray(image).astype('float64') / 255.0


class ImagePairMetric(ImagePairMixin, Metric):
    """Base class for full-reference image quality metrics."""

    def __init__(self, resize: bool = True) -> None:
        self.resize = resize

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        return [
            self._score_pair(*self._prepare_arrays(prediction, reference))
            for prediction, reference in zip(predictions, references)
        ]

    def _score_pair(self, prediction: Any, reference: Any) -> float:
        raise NotImplementedError


@register_metric(name=['psnr', 'PSNR'])
class PSNRScore(ImagePairMetric):
    """Peak signal-to-noise ratio for generated/reference image pairs."""

    def __init__(self, resize: bool = True, data_range: float = 1.0) -> None:
        super().__init__(resize=resize)
        if data_range <= 0:
            raise ValueError('data_range must be greater than 0.')
        self.data_range = data_range

    def _score_pair(self, prediction: Any, reference: Any) -> float:
        check_import('numpy', 'numpy', raise_error=True, feature_name='PSNR Metric')
        import numpy as np

        mse = float(np.mean((prediction - reference)**2))
        if mse == 0:
            return 100.0
        return float(20 * math.log10(self.data_range) - 10 * math.log10(mse))


@register_metric(name=['ssim', 'SSIM'])
class SSIMScore(ImagePairMetric):
    """Structural similarity for generated/reference image pairs."""

    def __init__(
        self,
        resize: bool = True,
        data_range: float = 1.0,
        window_size: int = 11,
        gaussian_weights: bool = True,
        sigma: float = 1.5,
        use_sample_covariance: bool = False,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        super().__init__(resize=resize)
        if data_range <= 0:
            raise ValueError('data_range must be greater than 0.')
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError('window_size must be an odd integer greater than or equal to 3.')
        if sigma <= 0:
            raise ValueError('sigma must be greater than 0.')
        if k1 <= 0 or k2 <= 0:
            raise ValueError('k1 and k2 must be greater than 0.')
        self.data_range = data_range
        self.window_size = window_size
        self.gaussian_weights = gaussian_weights
        self.sigma = sigma
        self.use_sample_covariance = use_sample_covariance
        self.k1 = k1
        self.k2 = k2

    def _score_pair(self, prediction: Any, reference: Any) -> float:
        check_import('numpy', 'numpy', raise_error=True, feature_name='SSIM Metric')
        import numpy as np

        if min(prediction.shape[:2]) < self.window_size:
            return self._global_ssim(prediction, reference)

        from numpy.lib.stride_tricks import sliding_window_view

        prediction_windows = sliding_window_view(prediction, (self.window_size, self.window_size), axis=(0, 1))
        reference_windows = sliding_window_view(reference, (self.window_size, self.window_size), axis=(0, 1))
        weights = self._window_weights()
        cov_norm = self._covariance_norm(weights)

        mu_prediction = (prediction_windows * weights).sum(axis=(-2, -1))
        mu_reference = (reference_windows * weights).sum(axis=(-2, -1))
        sigma_prediction = cov_norm * ((prediction_windows**2 * weights).sum(axis=(-2, -1)) - mu_prediction**2)
        sigma_reference = cov_norm * ((reference_windows**2 * weights).sum(axis=(-2, -1)) - mu_reference**2)
        sigma_cross = cov_norm * ((prediction_windows * reference_windows * weights).sum(axis=(-2, -1))
                                  - mu_prediction * mu_reference)

        ssim_map = self._ssim_from_moments(
            mu_prediction,
            mu_reference,
            sigma_prediction,
            sigma_reference,
            sigma_cross,
        )
        return float(np.mean(ssim_map))

    def _global_ssim(self, prediction: Any, reference: Any) -> float:
        check_import('numpy', 'numpy', raise_error=True, feature_name='SSIM Metric')
        import numpy as np

        mu_prediction = prediction.mean(axis=(0, 1))
        mu_reference = reference.mean(axis=(0, 1))
        cov_norm = self._global_covariance_norm(prediction.shape[0] * prediction.shape[1])
        sigma_prediction = cov_norm * prediction.var(axis=(0, 1))
        sigma_reference = cov_norm * reference.var(axis=(0, 1))
        sigma_cross = cov_norm * ((prediction - mu_prediction) * (reference - mu_reference)).mean(axis=(0, 1))
        return float(
            np.mean(
                self._ssim_from_moments(
                    mu_prediction,
                    mu_reference,
                    sigma_prediction,
                    sigma_reference,
                    sigma_cross,
                )
            )
        )

    def _window_weights(self) -> Any:
        check_import('numpy', 'numpy', raise_error=True, feature_name='SSIM Metric')
        import numpy as np

        if not self.gaussian_weights:
            weights = np.ones((self.window_size, self.window_size), dtype='float64')
        else:
            radius = self.window_size // 2
            coords = np.arange(-radius, radius + 1, dtype='float64')
            y_coords, x_coords = np.meshgrid(coords, coords, indexing='ij')
            weights = np.exp(-(x_coords**2 + y_coords**2) / (2 * self.sigma**2))
        weights = weights / weights.sum()
        return weights.reshape((1, 1, 1, self.window_size, self.window_size))

    def _covariance_norm(self, weights: Any) -> float:
        if not self.use_sample_covariance:
            return 1.0
        denominator = 1.0 - float((weights**2).sum())
        return 1.0 / denominator if denominator > 0 else 1.0

    def _global_covariance_norm(self, num_pixels: int) -> float:
        if not self.use_sample_covariance or num_pixels <= 1:
            return 1.0
        return num_pixels / (num_pixels - 1)

    def _ssim_from_moments(
        self,
        mu_prediction: Any,
        mu_reference: Any,
        sigma_prediction: Any,
        sigma_reference: Any,
        sigma_cross: Any,
    ) -> Any:
        c1 = (self.k1 * self.data_range)**2
        c2 = (self.k2 * self.data_range)**2
        numerator = (2 * mu_prediction * mu_reference + c1) * (2 * sigma_cross + c2)
        denominator = (mu_prediction**2 + mu_reference**2 + c1) * (sigma_prediction + sigma_reference + c2)
        return numerator / denominator


@register_metric(name=['lpips', 'LPIPS'])
class LPIPSScore(ImagePairMixin, SingletonMetric):
    """Learned perceptual image patch similarity for generated/reference image pairs."""

    def _init_once(
        self,
        net: str = 'alex',
        device: Optional[str] = None,
        resize: bool = True,
        spatial: bool = False,
    ) -> None:
        check_import(['lpips', 'torch'], ['lpips', 'torch'],
                     raise_error=True,
                     feature_name='LPIPS Metric',
                     extra='aigc')

        import lpips

        from evalscope.utils.model_utils import get_device

        self.resize = resize
        self.device = device or get_device()
        self.loss_fn = lpips.LPIPS(net=net, spatial=spatial).to(self.device)
        self.loss_fn.eval()

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        import torch

        scores = []
        for prediction, reference in zip(predictions, references):
            prediction_image, reference_image = self._prepare_pil_images(prediction, reference)
            prediction_tensor = self._pil_to_tensor(prediction_image).to(self.device)
            reference_tensor = self._pil_to_tensor(reference_image).to(self.device)
            with torch.no_grad():
                score = self.loss_fn(prediction_tensor, reference_tensor)
            scores.append(float(score.mean().detach().cpu().item()))
        return scores

    def _pil_to_tensor(self, image: Any) -> Any:
        check_import(['numpy', 'torch'], ['numpy', 'torch'], raise_error=True, feature_name='LPIPS Metric')
        import torch

        array = self._image_to_array(image).astype('float32') * 2 - 1
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


# ##################
# T2I Metrics ######
# ##################
@register_metric(name='VQAScore')
class VQAScore(T2IMetric):

    def _init_once(self, model: str = 'clip-flant5-xxl', device=None, cache_dir=None, **kwargs):
        device, cache_dir = self._resolve_defaults(device, cache_dir)
        from .t2v_metrics.vqascore import VQAScore
        self.model = VQAScore(model=model, device=device, cache_dir=cache_dir, **kwargs)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='PickScore')
class PickScore(T2IMetric):

    def _init_once(self, model: str = 'pickscore-v1', device=None, cache_dir=None, **kwargs):
        device, cache_dir = self._resolve_defaults(device, cache_dir)
        from .t2v_metrics.clipscore import CLIPScore
        self.model = CLIPScore(model=model, device=device, cache_dir=cache_dir, **kwargs)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='CLIPScore')
class CLIPScore(T2IMetric):

    def _init_once(self, model: str = 'openai:ViT-L-14-336', device=None, cache_dir=None, **kwargs):
        device, cache_dir = self._resolve_defaults(device, cache_dir)
        from .t2v_metrics.clipscore import CLIPScore
        self.model = CLIPScore(model=model, device=device, cache_dir=cache_dir, **kwargs)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='BLIPv2Score')
class BLIPv2Score(T2IMetric):

    def _init_once(self, model: str = 'blip2-itm', device=None, cache_dir=None, **kwargs):
        device, cache_dir = self._resolve_defaults(device, cache_dir)
        from .t2v_metrics.itmscore import ITMScore
        self.model = ITMScore(model=model, device=device, cache_dir=cache_dir, **kwargs)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='HPSv2Score')
class HPSv2Score(T2IMetric):

    def _init_once(self, model: str = 'hpsv2', device=None, cache_dir=None, **kwargs):
        device, cache_dir = self._resolve_defaults(device, cache_dir)
        from .t2v_metrics.clipscore import CLIPScore
        self.model = CLIPScore(model=model, device=device, cache_dir=cache_dir, **kwargs)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='HPSv2.1Score')
class HPSv2_1Score(T2IMetric):

    def _init_once(self, model: str = 'hpsv2.1', device=None, cache_dir=None, **kwargs):
        device, cache_dir = self._resolve_defaults(device, cache_dir)
        from .t2v_metrics.clipscore import CLIPScore
        self.model = CLIPScore(model=model, device=device, cache_dir=cache_dir, **kwargs)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='ImageRewardScore')
class ImageRewardScore(T2IMetric):

    def _init_once(self, model: str = 'image-reward-v1', device=None, cache_dir=None, **kwargs):
        device, cache_dir = self._resolve_defaults(device, cache_dir)
        from .t2v_metrics.itmscore import ITMScore
        self.model = ITMScore(model=model, device=device, cache_dir=cache_dir, **kwargs)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='FGA_BLIP2Score')
class FGA_BLIP2Score(T2IMetric):

    def _init_once(self, model: str = 'fga_blip2', device=None, cache_dir=None, **kwargs):
        device, cache_dir = self._resolve_defaults(device, cache_dir)
        from .t2v_metrics.itmscore import ITMScore
        self.model = ITMScore(model=model, device=device, cache_dir=cache_dir, **kwargs)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='MPS')
class MPS(T2IMetric):

    def _init_once(self, model: str = 'mps', device=None, cache_dir=None, **kwargs):
        device, cache_dir = self._resolve_defaults(device, cache_dir)
        from .t2v_metrics.clipscore import CLIPScore
        self.model = CLIPScore(model=model, device=device, cache_dir=cache_dir, **kwargs)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


# ##################
# Aggregators ######
# ##################
@register_aggregation(name='mean')
class Mean(Aggregator):

    name = 'mean'

    def agg_func(self, values: List[float]) -> float:
        return mean(values)

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate scores by computing the mean for each metric.

        Args:
            scores: List of sample scores to aggregate

        Returns:
            List of aggregated scores with mean values
        """
        if not scores:
            return []

        # Group score values by metric name
        metric_values = defaultdict(list)
        metric_sample_ids = defaultdict(list)

        for score in scores:

            for metric_name, value in score.score.value.items():
                metric_values[metric_name].append(value)
                metric_sample_ids[metric_name].append(score.sample_id)

        # Calculate mean for each metric
        aggregated_scores = []
        for metric_name, values in metric_values.items():
            if values:  # Only process non-empty value lists
                aggregated_scores.append(
                    AggScore(
                        score=self.agg_func(values),
                        metric_name=metric_name,
                        aggregation_name=self.name,
                        num=len(values),
                        ids=metric_sample_ids[metric_name]
                    )
                )

        return aggregated_scores


@register_aggregation(name='clipped_mean')
class ClippedMean(Mean):

    name = 'clipped_mean'

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def agg_func(self, values: List[float]) -> float:
        clipped_values = min(max(mean(values), self.clip_min), self.clip_max)
        return clipped_values


@register_aggregation(name='mean_and_pass_at_k')
class MeanPassAtK(Aggregator):

    def __init__(self):
        self.name = 'mean_and_pass_at_k'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Add per-metric pass@n for all n <= k to each sample, then mean-aggregate.

        For each metric:
        - Group scores by group_id
        - Collect binary correctness values
        - Infer k as (total samples / number of groups) assuming uniform repetitions
        - Compute per-group pass@n for all n from 1 to k via calculate_pass_at_k
        - Annotate each sample with metric_pass@n for its group (for all n)
        Finally run Mean() over the augmented metric set.
        """
        if not scores:
            return []

        # Extract metric names present in score values
        metrics = list(scores[0].score.value.keys())

        for metric_name in metrics:
            # group_id -> list[float] (0/1 correctness values)
            group_values: Dict[str, List[float]] = defaultdict(list)
            for s in scores:
                group_id = getattr(s, 'group_id', s.sample_id)
                value = float(s.score.value[metric_name])
                group_values[group_id].append(value)

            if not group_values:
                continue

            # Infer k (assumes roughly uniform repeats)
            k = int(len(scores) / len(group_values)) if len(group_values) > 0 else 1
            if k <= 0:
                k = 1

            # Prepare inputs for calculate_pass_at_k
            num_samples: List[int] = []
            num_correct: List[int] = []
            group_order: List[str] = []
            for gid, vals in group_values.items():
                group_order.append(gid)
                num_samples.append(len(vals))
                num_correct.append(int(sum(vals)))

            # Compute per-group pass@n for all n from 1 to k
            pass_at_n_maps = {}
            for n in range(1, k + 1):
                pass_at_n_list = calculate_pass_at_k(num_samples, num_correct, n)
                pass_at_n_maps[n] = {gid: float(v) for gid, v in zip(group_order, pass_at_n_list)}

            # Annotate each sample with its group's pass@n for all n
            for s in scores:
                group_id = getattr(s, 'group_id', s.sample_id)
                for n in range(1, k + 1):
                    s.score.value[f'{metric_name}_pass@{n}'] = pass_at_n_maps[n][group_id]

        # Delegate mean aggregation over original + injected pass@n metrics
        m = Mean()
        return m(scores)


@register_aggregation(name='mean_and_vote_at_k')
class MeanVoteAtK(Aggregator):

    def __init__(self):
        self.name = 'mean_and_vote_at_k'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate scores by computing vote@n for all n <= k for each metric using group_id.

        Vote@n selects the most frequent prediction among first n samples, then checks if
        that prediction is correct. This ensures vote@n has proper monotonicity properties.

        Note: vote@n computes accuracy per unique problem (one score per group_id), while
        mean_acc averages over all samples (including repeats). Therefore, vote@n can be
        higher or lower than mean_acc depending on sample ordering and repeat distribution.

        For each metric:
        - Group scores by group_id, preserving order
        - For each n from 1 to k, find most frequent prediction among first n samples
        - Check if most frequent prediction was ever marked correct (score=1.0) in those samples
        - Assign 1.0 if correct, 0.0 otherwise

        Args:
            scores: List of sample scores to aggregate

        Returns:
            List of aggregated scores with vote@n values for all n <= k
        """
        if not scores:
            return []

        # Freeze metric names before augmenting values
        metrics = list(scores[0].score.value.keys())

        for metric_name in metrics:
            # Group samples by group_id, preserving order
            # Store: (prediction, correctness_score)
            group_samples: Dict[str, List[tuple]] = defaultdict(list)
            for score in scores:
                group_id = getattr(score, 'group_id', score.sample_id)
                prediction = getattr(score.score, 'extracted_prediction', None)
                correctness = score.score.value[metric_name]
                group_samples[group_id].append((prediction, correctness))

            if not group_samples:
                continue

            # Calculate k as the repetition count
            k = int(len(scores) / len(group_samples)) if len(group_samples) > 0 else 1
            if k <= 0:
                k = 1

            # Compute vote@n for all n from 1 to k for each group
            vote_at_n_maps: Dict[int, Dict[str, float]] = {}
            for n in range(1, k + 1):
                vote_at_n_maps[n] = {}
                for group_id, samples in group_samples.items():
                    # Consider only first n samples for this group
                    n_samples = samples[:n]

                    # Count prediction frequencies
                    prediction_counts = defaultdict(int)
                    for prediction, _ in n_samples:
                        prediction_counts[prediction] += 1

                    # Select most frequent prediction (ties broken by first occurrence)
                    most_frequent_pred = max(prediction_counts, key=prediction_counts.get)

                    # Check if this prediction was ever correct in the first n samples
                    is_correct = any(
                        pred == most_frequent_pred and correctness == 1.0 for pred, correctness in n_samples
                    )

                    vote_at_n_maps[n][group_id] = 1.0 if is_correct else 0.0

            # Annotate each sample with its group's vote@n for all n
            for score in scores:
                group_id = getattr(score, 'group_id', score.sample_id)
                for n in range(1, k + 1):
                    score.score.value[f'{metric_name}_vote@{n}'] = vote_at_n_maps[n][group_id]

        # Calculate the mean value for all metrics and their corresponding vote@n
        m = Mean()
        return m(scores)


@register_aggregation(name='mean_and_pass_hat_k')
class MeanPassHatK(Aggregator):

    def __init__(self):
        self.name = 'mean_and_pass_hat_k'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Add per-metric pass^n for all n <= k using calculate_pass_hat_k, then mean-aggregate.

        For each metric:
        - Group scores by group_id
        - Collect binary correctness values
        - Infer k as approximate repeats and clamp to min attempts across groups
        - Compute per-group pass^n for all n from 1 to k via calculate_pass_hat_k
        - Annotate each sample with metric_pass^{n} for its group (for all n)
        Finally run Mean() over the augmented metric set.
        """
        if not scores:
            return []

        # Freeze metric names before augmenting values to avoid iterating injected keys
        metrics = list(scores[0].score.value.keys())

        for metric_name in metrics:
            # group_id -> list[float] (0/1 correctness values)
            group_values: Dict[str, List[float]] = defaultdict(list)
            for s in scores:
                group_id = getattr(s, 'group_id', s.sample_id)
                value = float(s.score.value[metric_name])
                group_values[group_id].append(value)

            if not group_values:
                continue

            # Infer repeats and clamp to the smallest group size to satisfy n <= min_n
            approx_k = int(len(scores) / len(group_values)) if len(group_values) > 0 else 1
            min_n = min(len(vals) for vals in group_values.values())
            k = max(1, min(approx_k, min_n))

            # Compute per-group pass^n for all n from 1 to k
            pass_hat_n_maps: Dict[int, Dict[str, float]] = {}
            for n in range(1, k + 1):
                pass_hat_n_maps[n] = {}
                for gid, vals in group_values.items():
                    total = len(vals)
                    correct = int(sum(vals))
                    pass_hat_n_maps[n][gid] = float(calculate_pass_hat_k(total, correct, n))

            # Annotate each sample with its group's pass^n for all n
            for s in scores:
                group_id = getattr(s, 'group_id', s.sample_id)
                for n in range(1, k + 1):
                    s.score.value[f'{metric_name}_pass^{n}'] = pass_hat_n_maps[n][group_id]

        # Mean aggregate over original + injected pass^n metrics
        m = Mean()
        return m(scores)
