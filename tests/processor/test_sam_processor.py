from loguru import logger
import pytest
import os
from pathlib import Path
from omegaconf import OmegaConf
from one_model.common import Config
from addict import Dict
from one_model.processor import SamPreProcessor
from one_model.common.mm_utils import load_image_as_ndarray

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_image_processor():
    test_img = f"{cur_dir}/../view.jpg"
    image = load_image_as_ndarray(test_img)
    processor = SamPreProcessor()
    image_tensor, image_shape = processor.process(image)
    logger.info(f"image_tensor.shape: {image_tensor.shape}")
    logger.info(f"image_shape: {image_shape}")
