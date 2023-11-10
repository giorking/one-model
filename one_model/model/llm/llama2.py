from loguru import logger
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)
from one_model.common.registries import LLM_MODEL_REGISTRY


class LLAMA2Config(LlamaConfig):
    model_type = "llama2"


@LLM_MODEL_REGISTRY.register(alias="llama2")
class LLAMA2Model(LlamaModel):
    config_class = LLAMA2Config

    def __init__(self, config: LlamaConfig):
        super(LLAMA2Model, self).__init__(config)


class LLAMA2ForCausalLM(LlamaForCausalLM):
    config_class = LLAMA2Config
