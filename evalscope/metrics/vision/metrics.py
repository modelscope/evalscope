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
