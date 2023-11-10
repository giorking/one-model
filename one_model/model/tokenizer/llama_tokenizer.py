import torch
import torch.nn as nn

from one_model.common.registries import TOKENIZER_REGISTRY
from transformers import AutoTokenizer
from loguru import logger


@TOKENIZER_REGISTRY.register(alias="llama2")
class LlamaTokenizer(object):
    def __init__(self, model_name_or_path, add_tokens: dict) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        # auto add tokens
        if add_tokens:
            for key, value in add_tokens.items():
                logger.info("try add token {}, value {}", key, value)
                token = value["token"]
                special_token = value.get("special_token", False)
                tokens = token.split(",")
                self._tokenizer.add_tokens(list(tokens), special_tokens=special_token)

    @property
    def tokenizer(self):
        return self._tokenizer

    def __len__(self):
        return len(self._tokenizer)

    @staticmethod
    def from_config(config: dict):
        model_name = config.get("model_name_or_path")
        add_tokens = config.get("add_tokens", None)
        return LlamaTokenizer(model_name, add_tokens)
