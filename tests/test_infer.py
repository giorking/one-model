from loguru import logger
import pytest
import os
from pathlib import Path
from omegaconf import OmegaConf
from one_model.common import Config
from addict import Dict
from one_model.inference import Infer


cur_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(cur_dir, "test_config_13B.yaml")


def test_generate_ids_13B():
    image = os.path.join(cur_dir, "view.jpg")
    prompt = "describe the pic"
    vis_save_path = "vis"
    config: Config = Config(Dict(cfg_path=config_path))
    infer = Infer(config)
    infer.init_model()
    logger.info("predict image {}, prompt {}", image, prompt)
    input_ids = infer.model.generate_input_ids(image, prompt, infer.model.device)
    logger.info("image sum count {}", (input_ids == -200).sum())

    prompt = "segment the lake"
    logger.info("predict image {}, prompt {}", image, prompt)
    input_ids = infer.model.generate_input_ids(image, prompt, infer.model.device)
    logger.info("image sum count {}", (input_ids == -200).sum())


def test_infer_13B():
    # image = os.path.join(cur_dir, "view.jpg")
    image = "/opt/product/one_model/resources/truck.jpg"
    prompt = "describe the pic"
    vis_save_path = "vis"
    config: Config = Config(Dict(cfg_path=config_path))
    infer = Infer(config)
    infer.init_model()
    logger.info("predict image {}, prompt {}", image, prompt)
    predict_text, image_path = infer.predict(image, prompt, vis_save_path)
    logger.info("predict text {}", predict_text)

    prompt = "segment the lake"
    prompt = "can you segment the left truck in the image?"
    logger.info("predict image {}, prompt {}", image, prompt)
    predict_text, image_path = infer.predict(image, prompt, vis_save_path)
    logger.info("predict text {}", predict_text)
