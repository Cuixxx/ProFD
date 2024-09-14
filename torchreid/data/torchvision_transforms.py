import math
import numpy as np
import random
from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
# from timm.data.random_erasing import RandomErasing
from torchreid.data.data_augmentation.random_erasing import RandomErasing
import albumentations as A
from torchreid.data.masks_transforms import masks_preprocess_all, AddBackgroundMask, ResizeMasks, PermuteMasksDim, \
    RemoveBackgroundMask

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        image = F.resize(image, self.size)
        if mask is not None:
            mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, mask


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if mask is not None:
            mask = F.crop(mask, *crop_params)
        return image, mask


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.center_crop(image, self.size)
        if mask is not None:
            mask = F.center_crop(mask, self.size)
        return image, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask

class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue, p):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.colojitter = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.prob = p

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = self.colojitter(image)
        return image, mask


class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, mask):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if mask is not None:
            mask = F.pad(mask, self.padding_n, self.padding_fill_target_value)
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask):
        image = F.to_tensor(image)
        # if mask is not None:
        #     mask = torch.as_tensor(np.array(mask))
        return image, mask


class Compose(object):
    def __init__(self, transforms, mask_transforms):
        self.transforms = transforms
        self.mask_transforms = mask_transforms
    def __call__(self, image, mask=None):
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        for t in self.transforms:
            image, mask = t(image, mask)
        for t in self.mask_transforms:
            result = t(image=image, mask=mask)
            image, mask = result['image'], result['mask']
        return {'image': image, 'mask': mask}

def build_transforms(
    height,
    width,
    config,
    mask_scale=4,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    remove_background_mask=False,
    masks_preprocess = 'none',
    softmax_weight = 0,
    mask_filtering_threshold = 0.3,
    background_computation_strategy = 'threshold',
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_tr = []


    if 'random_flip' in transforms or 'rf' in transforms:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip(flip_prob=0.5)]

    if 'random_crop' in transforms or 'rc' in transforms:
        print('+ random crop')
        pad_size = 10
        transform_tr += [Pad(padding_n=10, padding_fill_value=0, padding_fill_target_value=0),
                         RandomCrop([height, width])]

    if 'color_jitter' in transforms or 'cj' in transforms:
        print('+ color jitter')
        transform_tr += [
            ColorJitter(brightness=config.data.cj.brightness,
                        contrast=config.data.cj.contrast,
                        saturation=config.data.cj.saturation,
                        hue=config.data.cj.hue,
                        p=config.data.cj.p,
                        )
        ]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensor()]

    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in transforms or 're' in transforms:
        print('+ random erase')
        transform_tr += [RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu')]



    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = [
        ToTensor(),
        normalize,

    ]
    transform_tr_mask = []
    transform_te_mask = []
    if remove_background_mask:  # ISP masks
        print('+ use remove background mask')
        # remove background before performing other transforms
        transform_tr_mask = [RemoveBackgroundMask()] + transform_tr_mask
        transform_te_mask = [RemoveBackgroundMask()] + transform_te_mask

        # Derive background mask from all foreground masks once other tasks have been performed
        print('+ use add background mask')
        transform_tr_mask += [AddBackgroundMask('sum')]
        transform_te_mask += [AddBackgroundMask('sum')]
    else:  # Pifpaf confidence based masks
        if masks_preprocess != 'none':
            print('+ masks preprocess = {}'.format(masks_preprocess))
            masks_preprocess_transform = masks_preprocess_all[masks_preprocess]
            transform_tr_mask += [masks_preprocess_transform()]
            transform_te_mask += [masks_preprocess_transform()]

        print('+ use add background mask')
        transform_tr_mask += [AddBackgroundMask(background_computation_strategy, softmax_weight, mask_filtering_threshold)]
        transform_te_mask += [AddBackgroundMask(background_computation_strategy, softmax_weight, mask_filtering_threshold)]

    transform_tr_mask += [ResizeMasks(height, width, mask_scale)]
    transform_te_mask += [ResizeMasks(height, width, mask_scale)]

    transform_tr = Compose(transform_tr, transform_tr_mask)
    transform_te = Compose(transform_te, transform_te_mask)


    return transform_tr, transform_te


# def build_torch_transforms(
#     height,
#     width,
#     config,
#     mask_scale=4,
#     transforms='random_flip',
#     norm_mean=[0.485, 0.456, 0.406],
#     norm_std=[0.229, 0.224, 0.225],
#     remove_background_mask=False,
#     masks_preprocess = 'none',
#     softmax_weight = 0,
#     mask_filtering_threshold = 0.3,
#     background_computation_strategy = 'threshold',
#     **kwargs
# ):
#     train_transforms = T.Compose([
#         T.RandomHorizontalFlip(p=0.5),
#         T.Pad(10),
#         T.RandomCrop([height, width]),
#         T.ToTensor(),
#         T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu'),
#         # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
#     ])
#
#     val_transforms = T.Compose([
#         T.ToTensor(),
#         T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     return train_transforms, val_transforms



if __name__ == '__init__':
    transform = Compose([
        Resize(INPUT.IMG_SIZE),
        RandomHorizontalFlip(flip_prob=INPUT.PROB),
        Pad(INPUT.PADDING, 0, 0),
        RandomCrop(INPUT.IMG_SIZE),
        ToTensor(),
        Normalize(mean=INPUT.PIXEL_MEAN, std=INPUT.PIXEL_STD)
    ])
