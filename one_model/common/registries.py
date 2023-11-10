from loguru import logger
from .registry import Registry

"""
    registry hold all the registry, alias name -> alias implementation
"""

DATASET_REGISTRY = Registry("Dataset")
DATASET_REGISTRY.__doc__ = """
data init registry
"""

ENCODER_REGISTRY = Registry("Encoder")
ENCODER_REGISTRY.__doc__ = """
encoder registry
"""

DECODER_REGISTRY = Registry("Decoder")
DECODER_REGISTRY.__doc__ = """
encoder registry
"""

LLM_MODEL_REGISTRY = Registry("LLM")
LLM_MODEL_REGISTRY.__doc__ = """
llm model registry
"""

PROJECTOR_REGISTRY = Registry("Projector")
PROJECTOR_REGISTRY.__doc__ = """
projector registry
"""

TOKENIZER_REGISTRY = Registry("Tokenizer")
TOKENIZER_REGISTRY.__doc__ = """
tokenizer registry
"""
