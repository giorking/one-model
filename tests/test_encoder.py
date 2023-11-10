import pytest
from loguru import logger
from one_model.common.registries import ENCODER_REGISTRY
from one_model.dataset import *
from one_model.common.mm_utils import load_image
from addict import Dict
from one_model.dataset import *
from one_model.model.encoder import CLIPEncoder
from one_model.model.projector import *
from addict import Dict
from one_model.common.config import Config
from one_model.model.llm import *
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(cur_dir, "test_config_7B.yaml")


def test_clip_encoder():
    config: Config = Config(Dict(cfg_path=config_path))
    clip_encoder_large_config = config.encoder_cfg.clip_encoder
    clip_encoder_cls = ENCODER_REGISTRY.get(clip_encoder_large_config.type)
    clip_encoder: CLIPEncoder = clip_encoder_cls.from_config(clip_encoder_large_config)

    image_file = f"{cur_dir}/view.jpg"
    image_processor = clip_encoder.image_processor
    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ].cuda()
    logger.info(f"image_tensor: {image_tensor.shape}")
    image_features = clip_encoder(image_tensor)
    logger.info(f"image_features: {image_features.shape}")
