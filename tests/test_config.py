from loguru import logger
import pytest
import os
from pathlib import Path
from omegaconf import OmegaConf
from one_model.common import Config
from addict import Dict


cur_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(cur_dir, "test_config_7B.yaml")


def test_parse_config():
    default_cfg = OmegaConf.load(config_path)
    logger.info("default cfg {}", default_cfg)


def test_config():
    config: Config = Config(Dict(cfg_path=config_path))
    assert config.dataset_cfg is not None
    logger.info("config {}", config)
