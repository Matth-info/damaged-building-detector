# File describing some data augmentation pipeline using Albumentations

import albumentations as A
from albumentations.pytorch import ToTensorV2

training_transformations = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=15, p=0.5),
        A.GridDistortion(p=0.35),
        A.RandomCrop(384, 384, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussianBlur(p=0.25),
        ToTensorV2()
    ]
)
