import pytest
from loguru import logger
from one_model.common.registries import TOKENIZER_REGISTRY
from one_model.dataset import *
from one_model.common.mm_utils import load_image
from addict import Dict
from one_model.dataset import *
from one_model.model.tokenizer import LlamaTokenizer
from one_model.model.projector import *
from addict import Dict
from one_model.common.config import Config
from one_model.model.llm import *
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(cur_dir, "test_config_13B_train.yaml")


def test_add_tokenizer():
    config: Config = Config(Dict(cfg_path=config_path))
    tokenizer_cfg = config.tokenizer_cfg.llama2_13B
    tokenizer_cls = TOKENIZER_REGISTRY.get(tokenizer_cfg.type)
    tokenizer: LlamaTokenizer = tokenizer_cls.from_config(tokenizer_cfg)
    logger.info("tokenizer len {}", len(tokenizer))
    # test save tokenizer
    tokenizer.tokenizer.save_pretrained("/tmp/test_tokenizer")
