import pytest
from loguru import logger
from one_model.common.registries import PROJECTOR_REGISTRY
from one_model.dataset import *
from one_model.model.encoder.clip_encoder import CLIPEncoder
from one_model.model.projector import *
from addict import Dict
from one_model.common.config import Config
import os
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(cur_dir, "test_config_7B.yaml")


def test_liner_projector_7B():
    config: Config = Config(Dict(cfg_path=config_path))
    image_proj_7B_config = config.projector_cfg.image_proj_7B
    projector_cls = PROJECTOR_REGISTRY.get(image_proj_7B_config.type)
    logger.info("proj 7B config {}", image_proj_7B_config)
    projector = projector_cls.from_config(image_proj_7B_config)
    logger.info("projector {}", projector)

    input = torch.randn(1280, projector.in_dim)
    output = projector(input)
    logger.info(output.size())


def test_mlp_projector_7B():
    config: Config = Config(Dict(cfg_path=config_path))
    mlp_proj_7B_config = config.projector_cfg.mlp_out_7B
    projector_cls = PROJECTOR_REGISTRY.get(mlp_proj_7B_config.type)
    logger.info("proj 7B config {}", mlp_proj_7B_config)
    projector = projector_cls.from_config(mlp_proj_7B_config)
    logger.info("projector {}", projector)

    input = torch.randn(1280, projector.in_dim)
    output = projector(input)
    logger.info(output.size())


def test_dummy_projector():
    config: Config = Config(Dict(cfg_path=config_path))
    dummy_config = config.projector_cfg.dummy
    projector_cls = PROJECTOR_REGISTRY.get(dummy_config.type)
    projector = projector_cls.from_config(dummy_config)
    logger.info("projector {}", projector)

    input = torch.randn(1280, projector.in_dim)
    output = projector(input)
    logger.info(output.size())
