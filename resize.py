# import albumentations as A
import os
import jpeg4py as jpeg
import cv2
import numpy as np
import imageio
import torchvision.transforms as T
from PIL import Image
import torch

def resize_img(path):
    resize_path = path + '_resize'
    transform = T.Resize([256, 128], interpolation=3)
    if not os.path.exists(resize_path): 
        os.makedirs(resize_path)
    for name in os.listdir(path):
        if name.split('.')[-1] == 'jpg':
            # img = jpeg.JPEG(os.path.join(path, name)).decode()
            img = Image.open(os.path.join(path, name)).convert('RGB')
            img = transform(img)
            # imageio.imwrite(os.path.join(resize_path, name), result['image'])
            img.save(os.path.join(resize_path, name), lossless=True)


def resize_Occluded_img(path):
    resize_path = path + '_resize'
    transform = T.Resize([256, 128], interpolation=3)
    for sub_dir in os.listdir(path):
        temp_path = os.path.join(path, sub_dir)
        temp_resize_path = os.path.join(resize_path, sub_dir)
        if not os.path.exists(temp_resize_path):  
            os.makedirs(temp_resize_path)
        for name in os.listdir(temp_path):
            if name.split('.')[-1] == 'tif':
                # img = jpeg.JPEG(os.path.join(path, name)).decode()
                img = Image.open(os.path.join(temp_path, name)).convert('RGB')
                img = transform(img)
                # imageio.imwrite(os.path.join(resize_path, name), result['image'])
                img.save(os.path.join(temp_resize_path, name), lossless=True)

def resize_P_Duke_img(path):
    resize_path = path + '_resize'
    transform = T.Resize([256, 128], interpolation=3)
    for sub_dir in os.listdir(path):
        for subsub_dir in os.listdir(os.path.join(path, sub_dir)):
            temp_path = os.path.join(path, sub_dir, subsub_dir)
            temp_resize_path = os.path.join(resize_path, sub_dir, subsub_dir)
            if not os.path.exists(temp_resize_path):  
                os.makedirs(temp_resize_path)
            for name in os.listdir(temp_path):
                if name.split('.')[-1] == 'jpg':
                    # img = jpeg.JPEG(os.path.join(path, name)).decode()
                    img = Image.open(os.path.join(temp_path, name)).convert('RGB')
                    img = transform(img)
                    # imageio.imwrite(os.path.join(resize_path, name), result['image'])
                    img.save(os.path.join(temp_resize_path, name), lossless=True)

def resize_P_Duke_mask(path):
    resize_path = path + '_resize'
    transform = T.Resize([256, 128], interpolation=3)
    for sub_dir in os.listdir(path):
        temp_path = os.path.join(path, sub_dir)
        temp_resize_path = os.path.join(resize_path, sub_dir)
        if not os.path.exists(temp_resize_path):  
            os.makedirs(temp_resize_path)
        for name in os.listdir(temp_path):
            if name.split('.')[-1] == 'npy':
                masks = np.load(os.path.join(temp_path, name))
                # masks = np.transpose(masks, (1, 2, 0))
                masks = torch.Tensor(masks)
                masks = transform(masks)
                # masks = np.transpose(masks, (2, 0, 1))
                np.save(os.path.join(temp_resize_path, name), masks)

def resize_Occluded_mask(path):
    resize_path = path + '_resize'
    transform = T.Resize([256, 128], interpolation=3)
    if not os.path.exists(resize_path):  
        os.makedirs(resize_path)
    for name in os.listdir(path):
        if name.split('.')[-1] == 'npy':
            masks = np.load(os.path.join(path, name))
            # masks = np.transpose(masks, (1, 2, 0))
            masks = torch.Tensor(masks)
            masks = transform(masks)
            # masks = np.transpose(masks, (2, 0, 1))
            np.save(os.path.join(resize_path, name), masks)

def resize_mask(path):
    resize_path = path + '_resize'
    transform = T.Resize([256, 128], interpolation=3)
    if not os.path.exists(resize_path):  
        os.makedirs(resize_path)
    for name in os.listdir(path):
        if name.split('.')[-1] == 'npy':
            masks = np.load(os.path.join(path, name))
            masks = np.transpose(masks, (1, 2, 0))
            masks = torch.Tensor(masks)
            result = transform(masks)
            masks = np.transpose(masks, (2, 0, 1))
            np.save(os.path.join(resize_path, name), masks)

if __name__ == '__main__':
    path = '/ReID_datasets/Occluded_REID/masks/pifpaf_maskrcnn_filtering/whole_body_images'
    resize_Occluded_mask(path)
    path = '/ReID_datasets/Occluded_REID/masks/pifpaf_maskrcnn_filtering/occluded_body_images'
    resize_Occluded_mask(path)