import cv2
import torchvision.transforms as transforms
from albumentations import (
    ChannelShuffle,
    ColorJitter,
    Compose,
    Cutout,
    ElasticTransform,
    Equalize,
    GaussianBlur,
    GaussNoise,
    GridDistortion,
    HorizontalFlip,
    Normalize,
    OpticalDistortion,
    RandomGridShuffle,
    RandomResizedCrop,
    Solarize,
    SomeOf,
    ToGray,
)
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.pytorch.transforms import ToTensorV2


def get_groupaugment_transformation(
    p_color_transformations,
    p_geometric_transformations,
    p_non_rigid_transformations,
    p_quality_transformations,
    p_exotic_transformations,
    n_color_transformations,
    n_geometric_transformations,
    n_non_rigid_transformations,
    n_quality_transformations,
    n_exotic_transformations,
    n_total,
    normalize,
    crop_size,
    crop_scale
):
    transform = Compose(
        [
            # basic SSL transformation
            RandomResizedCrop(
                height=crop_size, width=crop_size, scale=crop_scale, interpolation=cv2.INTER_CUBIC
            ),
            SomeOf(
                [
                    # color transformations
                    SomeOf(
                        [
                            ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=1
                            ),
                            ToGray(p=1),
                            Solarize(p=1),
                            Equalize(p=1),
                            ChannelShuffle(p=1),
                        ],
                        n=n_color_transformations,
                        replace=False,
                        p=p_color_transformations,
                    ),
                    # geometric transformations
                    SomeOf(
                        [
                            ShiftScaleRotate(interpolation=cv2.INTER_CUBIC, p=1),
                            HorizontalFlip(p=1),
                        ],
                        n=n_geometric_transformations,
                        replace=False,
                        p=p_geometric_transformations,
                    ),
                    # non-rigid transformations
                    SomeOf(
                        [
                            ElasticTransform(
                                alpha=0.5,
                                sigma=10,
                                alpha_affine=5,
                                interpolation=cv2.INTER_CUBIC,
                                p=1,
                            ),
                            GridDistortion(interpolation=cv2.INTER_CUBIC, p=1),
                            OpticalDistortion(
                                distort_limit=0.5,
                                shift_limit=0.5,
                                interpolation=cv2.INTER_CUBIC,
                                p=1,
                            ),
                        ],
                        n=n_non_rigid_transformations,
                        replace=False,
                        p=p_non_rigid_transformations,
                    ),
                    # quality transformations
                    SomeOf(
                        [
                            GaussianBlur(p=1),
                            GaussNoise(p=1),
                        ],
                        n=n_quality_transformations,
                        replace=False,
                        p=p_quality_transformations,
                    ),
                    # exotic transformations
                    SomeOf(
                        [
                            RandomGridShuffle(p=1),
                            Cutout(num_holes=4, p=1),
                        ],
                        n=n_exotic_transformations,
                        replace=False,
                        p=p_exotic_transformations,
                    ),
                ],
                n=n_total,
                replace=False,
                p=1,
            ),
            Normalize(normalize[0], normalize[1]),
            ToTensorV2(),
        ],
        p=1,
    )

    return transform

