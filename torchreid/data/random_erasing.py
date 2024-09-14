from albumentations import CoarseDropout
import numpy as np
import torch
from typing import Iterable,Tuple
from albumentations.augmentations.dropout.functional import cutout
class CustomCoarseDropout(CoarseDropout):
    def __init__(self, *args, **kwargs):
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)

    def apply(self, img, fill_value=(0, 0, 0), holes: Iterable[Tuple[int, int, int, int]] = (), **params):
        # 在这里添加你的自定义逻辑
        # 你可以使用父类的 apply 方法来应用原始的 CoarseDropout 效果
        # img = super().apply(img, fill_value=fill_value, **params)
        x1, y1, x2, y2 = holes[0]
        h,w = y2-y1, x2-x1
        fill_value = torch.empty((h, w, 3), device='cpu').normal_()
        return cutout(img, holes, fill_value)


if __name__ == '__main__':
    # 示例用法
    custom_dropout = CustomCoarseDropout(
        min_holes=1,
        max_holes=1,
        min_height=10,
        max_height=50,
        min_width=10,
        max_width=50,
        fill_value=(0, 0, 0),
        custom_param1=your_custom_value1,
        custom_param2=your_custom_value2,
        p=0.5,
    )

    # 应用到图像
    image_path = "path_to_your_image.jpg"
    image = Image.open(image_path).convert("RGB")
    transformed_image = custom_dropout(image=np.array(image))['image']

