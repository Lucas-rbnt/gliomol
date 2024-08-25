# Standard libraries
from typing import Hashable, Mapping

# Third-party libraries
import numpy as np
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, RandRotated, RandSpatialCropd, RandFlipd, NormalizeIntensityd, RandCoarseDropoutd, CenterSpatialCropd
from monai.transforms.transform import Transform, MapTransform
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor
from monai.transforms.utils_pytorch_numpy_unification import percentile, clip
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import DtypeLike, KeysCollection
from monai.data.meta_obj import get_track_meta
from monai.utils.enums import TransformBackends


class PercentileClipper(Transform):
    
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, lower: float, upper, channel_wise: bool = False, nonzero: bool = False) -> None:
        if not isinstance(lower, (int, float)):
            raise ValueError(f"lower must be a float or int number, got {type(lower)} {lower}.")
        if not isinstance(upper, (int, float)):
            raise ValueError(f"upper must be a float or int number, got {type(upper)} {upper}.")
        self.lower = lower
        self.upper = upper
        self.channel_wise = channel_wise
        self.nonzero = nonzero

    def _clipper(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img, *_ = convert_data_type(img, dtype=torch.float32)
        
        if self.nonzero:
            slices = img != 0
        else:
            if isinstance(img, np.ndarray):
                slices = np.ones_like(img, dtype=bool)
            else:
                slices = torch.ones_like(img, dtype=torch.bool)
        if not slices.any():
            return img
        
        a_min: float = percentile(img[slices], self.lower)  # type: ignore
        a_max: float = percentile(img[slices], self.upper)  # type: ignore

        img[slices] = clip(img[slices], a_min, a_max)
        return img
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t = convert_to_tensor(img, track_meta=False)
        if self.channel_wise:
            img_t = torch.stack([self._clipper(img=d) for d in img_t])  # type: ignore
        else:
            img_t = self._clipper(img=img_t)

        return convert_to_dst_type(img_t, dst=img)[0]


class PercentileClipperd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend: the amount to subtract by (usually the mean)
        divisor: the amount to divide by (usually the standard deviation)
        nonzero: whether only normalize non-zero values.
        channel_wise: if True, calculate on each channel separately, otherwise, calculate on
            the entire image directly. default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = PercentileClipper.backend

    def __init__(
        self,
        keys: KeysCollection,
        lower: float,
        upper: float,
        nonzero: bool = False,
        channel_wise: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.clipper = PercentileClipper(lower=lower, upper=upper, nonzero=nonzero, channel_wise=channel_wise)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.clipper(d[key])
        return d


train_transforms_regular = Compose([
    LoadImaged(keys='image', image_only=True),
    EnsureChannelFirstd(keys='image'),
    Orientationd(keys='image', axcodes="RAS"),
    PercentileClipperd(keys='image', lower=1, upper=99, nonzero=True, channel_wise=True),
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    CenterSpatialCropd(keys='image', roi_size=(128, 128, 128)),
    RandRotated(keys='image', range_x=np.pi/9, prob=0.3),
    RandCoarseDropoutd(keys='image', prob=0.9, holes=8, spatial_size=12, dropout_holes=True, max_spatial_size=20),
])

val_transforms_regular = Compose([
    LoadImaged(keys='image', image_only=True),
    EnsureChannelFirstd(keys='image'),
    Orientationd(keys='image', axcodes="RAS"),
    PercentileClipperd(keys='image', lower=1, upper=99, nonzero=True, channel_wise=True),
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    CenterSpatialCropd(keys='image', roi_size=(128, 128, 128)),
])

transforms_tumor_train = Compose([
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    RandRotated(keys='image', range_x=np.pi/9, prob=0.3),
    RandCoarseDropoutd(keys='image', prob=0.9, holes=4, spatial_size=6, dropout_holes=True, max_spatial_size=10),
                           
                                  ])

transforms_tumor_val = Compose([
    NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
                                  ])