from segment_anything.utils.transforms import ResizeLongestSide
import torch
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class SamPreProcessor:
    """sam 图片预处理"""

    def __init__(self, img_size=1024) -> None:
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.img_size = img_size
        # resize to longest side
        self.transform = ResizeLongestSide(self.img_size)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def process(self, image: np.ndarray):
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize_shape = image.shape[:2]
        image_tensor = self.preprocess(
            torch.from_numpy(image).permute(2, 0, 1).contiguous()
        )
        return image_tensor, resize_shape
