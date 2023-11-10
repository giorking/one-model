import pytest
from loguru import logger
from one_model.common.registries import DATASET_REGISTRY
from one_model.dataset import *
import transformers
from one_model.common.mm_utils import get_model_name_from_path
from addict import Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)


def test_one_model_load():
    ...
