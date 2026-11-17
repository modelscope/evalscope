import base64
import binascii
import math
from typing import Any, List, Optional, Tuple

from evalscope.api.metric import Metric, SingletonMetric, T2IMetric
from evalscope.api.registry import register_metric
from evalscope.utils.import_utils import check_import

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

        from evalscope.utils.uri_utils import file_as_data

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

        mse = float(np.mean((prediction - reference) ** 2))
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
        sigma_cross = cov_norm * (
            (prediction_windows * reference_windows * weights).sum(axis=(-2, -1)) - mu_prediction * mu_reference
        )

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
        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2
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
        check_import(
            ['lpips', 'torch'], ['lpips', 'torch'], raise_error=True, feature_name='LPIPS Metric', extra='aigc'
        )

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


# ############################
# Text Rendering Metrics #####
# ############################
_OCR_INSTANCES: dict = {}
"""Module-level OCR cache so the two CVTG scorers share one PaddleOCR engine per lang."""


def _get_cvtg_ocr(lang: str):
    if lang not in _OCR_INSTANCES:
        check_import('paddleocr', raise_error=True, feature_name='CVTG-2K Metric')
        from paddleocr import PaddleOCR

        _OCR_INSTANCES[lang] = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return _OCR_INSTANCES[lang]


def _recognize_cvtg_words(image_path: str, gt_words: List[str], backend: str, lang: str) -> List[str]:
    """Recognize the rendered words of an image; ``backend='stub'`` echoes the ground truth."""
    if backend == 'stub':
        return list(gt_words)
    result = _get_cvtg_ocr(lang).ocr(image_path, cls=True) or []
    words: List[str] = []
    for line in result[0] or []:
        words.extend(str(line[1][0]).lower().split())
    return words


class _CountsStashMixin:
    """Keep the per-image ``(numerator, denominator)`` of the last scoring call.

    The adapters re-aggregate corpus-level (official protocol: total numerator over total
    denominator), so the counts must survive the per-image ``apply()`` call. Keyed by image
    path: concurrent workers score distinct images, and the adapter reads the entry for the
    image it just scored. Bounded by the dataset size (two floats per image).
    """

    def _init_counts(self):
        self._counts_by_image: dict = {}

    def _stash_counts(self, image_path: str, numerator, denominator):
        self._counts_by_image[image_path] = (numerator, denominator)

    def counts_for(self, image_path: str):
        """Return ``(numerator, denominator)`` recorded for an image, or ``None``."""
        return self._counts_by_image.get(image_path)


@register_metric(name='CVTGWordAccuracy')
class CVTGWordAccuracy(_CountsStashMixin, T2IMetric):
    """CVTG-2K Word Accuracy: each ground-truth word counts when it appears in the
    recognized words (repeated ground-truth words are not consumed), matching the
    official TextCrafter evaluator.

    The OCR backend loads lazily; ``backend='stub'`` echoes the ground-truth words and
    exists for pipeline tests.
    """

    def _init_once(self, backend: str = 'paddleocr', lang: str = 'en', **kwargs):
        self.backend = backend
        self.lang = lang
        self._init_counts()

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        from evalscope.metrics.vision.text_render import extract_quoted_words, word_accuracy_counts

        scores = []
        for image_path, prompt in zip(images, texts):
            gt_words = extract_quoted_words(prompt)
            score, matched, total = word_accuracy_counts(
                gt_words, _recognize_cvtg_words(image_path, gt_words, self.backend, self.lang)
            )
            self._stash_counts(image_path, matched, total)
            scores.append(score)
        return scores


@register_metric(name='CVTGNED')
class CVTGNED(_CountsStashMixin, T2IMetric):
    """CVTG-2K NED: word-level best-match normalized Levenshtein distance (higher is better).

    Shares the OCR engine with :class:`CVTGWordAccuracy` and follows the same ground-truth
    extraction from the prompt's quoted fragments. The corpus aggregate is the similarity
    sum over all ground-truth words, so the stash keeps ``sum(similarities), word_count``.
    """

    def _init_once(self, backend: str = 'paddleocr', lang: str = 'en', **kwargs):
        self.backend = backend
        self.lang = lang
        self._init_counts()

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        from evalscope.metrics.vision.text_render import extract_quoted_words, word_ned_counts

        scores = []
        for image_path, prompt in zip(images, texts):
            gt_words = extract_quoted_words(prompt)
            score, sim_sum, word_count = word_ned_counts(
                gt_words, _recognize_cvtg_words(image_path, gt_words, self.backend, self.lang)
            )
            self._stash_counts(image_path, sim_sum, word_count)
            scores.append(score)
        return scores


@register_metric(name='LongTextScore')
class LongTextScore(_CountsStashMixin, T2IMetric):
    """LongText-Bench Text Score: Counter-intersection match of rendered text.

    Declares ``image_pair_metric`` so the adapter passes the ground-truth text (the
    sample target) instead of the prompt. The VLM transcription backend loads lazily;
    ``backend='stub'`` echoes the ground-truth text and exists for pipeline tests.

    ``lang=None`` (default) detects the language per sample from the ground-truth text
    (CJK -> ``zh``), so one run covers both languages; pass ``lang='en'``/``'zh'`` to
    force the official single-language mode.
    """

    image_pair_metric = True

    RECOGNIZE_PROMPT = (
        'Recognize the text in the image, only reply with the text content, but avoid '
        'repeating previously mentioned content. If no text is recognized, please reply '
        "with 'No text recognized'."
    )

    def _init_once(
        self, backend: str = 'qwen_vl', model: str = 'Qwen/Qwen2.5-VL-7B-Instruct', lang: str = None, **kwargs
    ):
        self.backend = backend
        self.model_name = model
        self.lang = lang
        self._vlm = None
        self._init_counts()

    def _transcribe(self, image_path: str, gt_text: str) -> str:
        if self.backend == 'stub':
            # Echo the ground truth so pipeline tests exercise the full scoring path.
            return gt_text
        if self._vlm is None:
            check_import(['transformers', 'qwen_vl_utils'], raise_error=True, feature_name='LongText-Bench Metric')
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self._vlm = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name, torch_dtype='auto', device_map='auto'
                ),
                AutoProcessor.from_pretrained(self.model_name),
            )
        vlm_model, processor = self._vlm
        import base64
        from io import BytesIO

        from PIL import Image

        with Image.open(image_path) as pil_image:
            buffered = BytesIO()
            pil_image.save(buffered, format='PNG')
            image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'image': f'data:image/png;base64,{image_b64}',
                    },
                    {
                        'type': 'text',
                        'text': self.RECOGNIZE_PROMPT,
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[Image.open(image_path)], return_tensors='pt').to(vlm_model.device)
        output = vlm_model.generate(**inputs, max_new_tokens=1024)
        generated = output[:, inputs.input_ids.shape[1] :]
        return processor.batch_decode(generated, skip_special_tokens=True)[0]

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        from evalscope.metrics.vision.text_render import text_score_counts

        scores = []
        for image_path, gt_text in zip(images, texts):
            score, matched, total = text_score_counts(gt_text, self._transcribe(image_path, gt_text), self.lang)
            self._stash_counts(image_path, matched, total)
            scores.append(score)
        return scores
