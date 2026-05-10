import albumentations as A
from albumentations.pytorch import ToTensorV2

NORM_MEAN = [0.485, 0.456, 0.406, 0.5]
NORM_STD  = [0.229, 0.224, 0.225, 0.25]

def get_train_transforms(image_size=512):
    return A.Compose([
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=512):
    return A.Compose([
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])

def get_tta_transforms(image_size=512):
    base = [
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base),
    ]